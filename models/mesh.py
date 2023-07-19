import torch
import torch.nn as nn
import numpy as np
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.renderer import RasterizationSettings, TexturesVertex, MeshRasterizer
from hashlib import sha1

from models.utils import PositionEncoding

class DNMPs(nn.Module):
    def __init__(self,
                 config,
                 voxel_centers,
                 voxel_size,
                 device,
                 num_faces=1,
                 mesh_embedding_dim=8):
        super(DNMPs, self).__init__()

        self.device = device
        self.config = config
        # self.decoder_fn = decoder_fn
        self.voxel_centers = nn.Parameter(voxel_centers, requires_grad=False)
        self.voxel_centers_offset = nn.Parameter(torch.zeros_like(voxel_centers), requires_grad=True)
        self.voxel_size = voxel_size
        num_voxels = voxel_centers.shape[0]
        print(f'Optimize {num_voxels} voxels')

        template_embedding = torch.tensor([0.7372,  0.3196, -0.1589, -0.1727,  0.4183, -0.1062,  0.1665, -0.2920], device=device).float() # template embedding, corresponding to sphere mesh
        self.template_embedding = template_embedding.clone()
        template_embedding = template_embedding.view(1,mesh_embedding_dim).repeat(voxel_centers.shape[0], 1)
        self.mesh_embeddings = nn.Parameter(template_embedding, requires_grad=True)
        self.template_verts, self.template_faces = self.get_template_faces()
        self.scale = voxel_size * 0.5 * 1.732
        self.num_faces = num_faces
        self.num_verts_per_mesh = self.template_verts.shape[0]
        self.num_meshes = voxel_centers.shape[0]
    
    def get_template_faces(self):
        template_mesh = ico_sphere(level=1, device=self.device)
        verts_packed = template_mesh.verts_packed()
        faces_packed = template_mesh.faces_packed()
        return verts_packed.squeeze(), faces_packed.squeeze()
    
    def render_depth_and_normal(self, cameras, sampled_idx, pix_coords, decoder_fn=None, img_hw=None):
        '''
            cameras:
            sampled_idx: [N]
            pix_coords: [M]
        '''

        if self.config.bin_size > 0:
            bin_size = self.config.bin_size
        else:
            max_image_size = max(int(img_hw[0]), int(img_hw[1]))
            bin_size=int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))
        
        num_faces_all = self.num_meshes * self.template_faces.shape[0]

        if self.config.max_faces_per_bin_factor > 0:
            max_faces_per_bin = num_faces_all // self.config.max_faces_per_bin_factor # May need to be tuned to avoid incomplete rendering results
        else:
            max_faces_per_bin = num_faces_all // 5

        rasterizer_settings = RasterizationSettings(
            image_size=(int(img_hw[0]),int(img_hw[1])),
            blur_radius=0., 
            faces_per_pixel=1,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=rasterizer_settings
        )

        mesh_embeddings = self.mesh_embeddings[sampled_idx.long()]
        mesh_embeddings = mesh_embeddings / (torch.norm(mesh_embeddings, dim=1, keepdim=True) + 1e-8)
        
        verts = decoder_fn(mesh_embeddings) # [N,V,3]
        faces = self.template_faces.unsqueeze(0).repeat(verts.shape[0],1,1) # [F,3]
        voxel_centers = self.voxel_centers + self.voxel_centers_offset
        voxel_centers = voxel_centers[sampled_idx.long()]
        verts = voxel_centers.unsqueeze(1) + verts * self.scale # [N,V,3]

        all_meshes = Meshes(verts, faces)
        join_meshes = join_meshes_as_scene(all_meshes)
        fragments = rasterizer(join_meshes)

        zbuf = fragments.zbuf
        zbuf = zbuf.view(-1,self.num_faces,1)
        zbuf = zbuf[:,:1,...]
        zbuf = zbuf[pix_coords]
        depth = zbuf.squeeze()

        pix_to_face = (fragments.pix_to_face.view(-1, self.num_faces))[pix_coords] # [num_rays, num_faces]
        bary_coords = (fragments.bary_coords.view(-1, self.num_faces, 3))[pix_coords] # [num_rays, num_faces, 3]
        pix_to_face = pix_to_face.view(1,pix_coords.shape[0],1,self.num_faces)
        bary_coords = bary_coords.view(1,pix_coords.shape[0],1,self.num_faces,3)
        verts_normals_packed = join_meshes.verts_normals_packed()
        faces = join_meshes.faces_packed()
        faces_verts_normals = verts_normals_packed[faces]
        sampled_normals = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts_normals)
        sampled_normals = sampled_normals.squeeze().view(-1, 3) # [num_rays, 3]
        
        return verts, depth, sampled_normals, join_meshes

class DNMPScene(nn.Module):
    def __init__(self,
                 config,
                 voxel_centers,
                 voxel_size,
                 mesh_embeddings,
                 device,
                 num_faces,
                 decoder_fn,
                 vertex_embedding_dim=16):
        super(DNMPScene, self).__init__()

        self.config = config
        self.device = device
        self.decoder_fn = decoder_fn

        # cache rasterizer results to speed up training;
        # this will enlarge the memory usage
        # disable this if you have limited memory
        self.use_rasterizer_cache = config.use_rasterizer_cache 
        if self.use_rasterizer_cache:
            self.rasterize_cache = {}
        
        mesh_ae_scale = voxel_size * 0.5 * 1.732
        verts = self.initialize_mesh_vertices(voxel_centers, mesh_embeddings, mesh_ae_scale)
        _, faces = self.get_faces()
        faces = faces.unsqueeze(0).repeat(verts.shape[0], 1, 1).to(verts)
        
        meshes = Meshes(verts.detach(), faces.detach())
        self.meshes = join_meshes_as_scene(meshes)
        self.verts = verts.data
        self.faces = faces.data
        
        verts_packed = meshes.verts_packed()
        pos_encoder = PositionEncoding(3, N_freqs=3)
        initial_verts_embedding = pos_encoder(verts_packed)
        self.vertex_embeddings = nn.Parameter(initial_verts_embedding.float().to(self.device), requires_grad=True)

        self.num_verts_per_mesh = verts.shape[1]
        self.num_meshes = voxel_centers.shape[0]
        self.num_faces = num_faces
    
    def initialize_mesh_vertices(self, voxel_centers, mesh_embeddings, mesh_ae_scale):
        chunk_size = 4096
        num_chunks = mesh_embeddings.shape[0] // chunk_size + 1
        verts_list = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = (chunk_idx+1) * chunk_size
            chunk_embeddings = mesh_embeddings[chunk_start:chunk_end]
            chunk_vertices = self.decoder_fn(chunk_embeddings)
            verts_list.append(chunk_vertices)
        verts = torch.cat(verts_list, dim=0)
        verts = verts * mesh_ae_scale + voxel_centers[:, None, :]
        return verts
    
    def get_faces(self):
        template_mesh = ico_sphere(level=1, device=self.device)
        verts_packed = template_mesh.verts_packed()
        faces_packed = template_mesh.faces_packed()
        return verts_packed.squeeze(), faces_packed.squeeze()
    
    def render_rays(self, cameras, pix_coords, img_hw=None, sampled_idx=None):
        '''
            cameras
            sampled_idx: [num_voxels]
            pix_coords: [num_rays]
        '''
        
        if sampled_idx is None:
            meshes = self.meshes
            verts_features_packed = self.vertex_embeddings
        else:
            vertex_embeddings = self.vertex_embeddings.view(self.num_meshes, self.num_verts_per_mesh, -1)
            textures = TexturesVertex(vertex_embeddings[sampled_idx])
            meshes = Meshes(self.verts[sampled_idx], self.faces[sampled_idx], textures)
            meshes = join_meshes_as_scene(meshes)
            verts_features_packed = meshes.textures.verts_features_packed()
        
        transform = cameras.get_world_to_view_transform().get_matrix()
        transform_hash = sha1(transform.cpu().numpy()).hexdigest()

        if (transform_hash in self.rasterize_cache) and self.use_rasterizer_cache:
            pix_to_face, bary_coords, zbuf = self.rasterize_cache[transform_hash]
            pix_to_face = torch.from_numpy(pix_to_face).long().to(self.device)
            bary_coords = torch.from_numpy(bary_coords).float().to(self.device)
            zbuf = torch.from_numpy(zbuf).float().to(self.device)
        
        else:
            if self.config.bin_size > 0:
                bin_size = self.config.bin_size
            else:
                max_image_size = max(int(img_hw[0]), int(img_hw[1]))
                bin_size=int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))
            
            if self.config.max_faces_per_bin_factor > 0:
                max_faces_per_bin = meshes._F // self.config.max_faces_per_bin_factor # May need to be tuned to avoid incomplete rendering results
            else:
                max_faces_per_bin = meshes._F // 5

            rasterizer_settings = RasterizationSettings(
                image_size=(int(img_hw[0]),int(img_hw[1])),
                blur_radius=0., 
                faces_per_pixel=self.num_faces,
                bin_size=bin_size,
                max_faces_per_bin=max_faces_per_bin
            )

            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=rasterizer_settings
            )

            with torch.no_grad():
                fragments = rasterizer(meshes)
                pix_to_face = fragments.pix_to_face
                bary_coords = fragments.bary_coords
                zbuf = fragments.zbuf
                if self.training and self.use_rasterizer_cache:
                    self.rasterize_cache[transform_hash] = (pix_to_face.data.cpu().numpy(), bary_coords.data.cpu().numpy(), zbuf.data.cpu().numpy())

        verts_packed = meshes.verts_packed()
        faces = meshes.faces_packed()
        faces_verts_features = verts_features_packed[faces]
        faces_verts = verts_packed[faces]

        pix_to_face = (pix_to_face.view(-1, self.num_faces))[pix_coords] # [num_rays, num_faces]
        bary_coords = (bary_coords.view(-1, self.num_faces, 3))[pix_coords] # [num_rays, num_faces, 3]

        pix_to_face_closest = pix_to_face[:,:1] # [num_rays]
        pix_to_face_closest = pix_to_face_closest.expand_as(pix_to_face)
        bary_coords_closest = bary_coords[:,:1,:] # [num_rays, 3]
        bary_coords_closest = bary_coords_closest.expand_as(bary_coords)
        pix_to_face_ = pix_to_face.clone()
        pix_to_face_bary = pix_to_face_.unsqueeze(-1).repeat(1,1,3)

        pix_to_face[pix_to_face_==-1] = pix_to_face_closest[pix_to_face_==-1]
        bary_coords[pix_to_face_bary==-1] = bary_coords_closest[pix_to_face_bary==-1]

        pix_to_face = pix_to_face.view(1,pix_coords.shape[0],1,self.num_faces)
        bary_coords = bary_coords.view(1,pix_coords.shape[0],1,self.num_faces,3)

        sampled_texture = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts_features)
        verts_normals_packed = meshes.verts_normals_packed()
        faces_verts_normals = verts_normals_packed[faces]
        sampled_normals = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts_normals)
        sampled_pts = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts)

        zbuf = zbuf.view(-1,self.num_faces)
        zbuf = zbuf[pix_coords]
        depth = zbuf # [num_rays, num_faces]
        depth_closest = depth[...,:1]
        depth_closest = depth_closest.expand_as(depth)
        depth[pix_to_face_==-1] = depth_closest[pix_to_face_==-1] # [num_rays, num_faces]

        sampled_texture = sampled_texture.view(depth.shape[0],self.num_faces,-1) # [num_rays, num_faces, num_channels]
        sampled_normals = sampled_normals.view(depth.shape[0],self.num_faces,3) # [num_rays, num_faces, 3]
        sampled_pts = sampled_pts.view(depth.shape[0],self.num_faces,3) # [num_rays, num_faces, 3]

        sorted_depth, sorted_indices = torch.sort(depth, dim=1) # [num_rays, num_faces]
        sorted_indices_texture = sorted_indices.unsqueeze(-1).repeat(1,1,sampled_texture.shape[-1])
        sorted_texture = torch.gather(sampled_texture, dim=1, index=sorted_indices_texture)
        sorted_indices_normals = sorted_indices.unsqueeze(-1).repeat(1,1,3)
        sorted_normals = torch.gather(sampled_normals, dim=1, index=sorted_indices_normals)
        sorted_indices_pts = sorted_indices.unsqueeze(-1).repeat(1,1,3)
        sorted_pts = torch.gather(sampled_pts, dim=1, index=sorted_indices_pts)

        return sorted_texture, sorted_pts, sorted_depth, sorted_normals