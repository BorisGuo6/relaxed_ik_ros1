import numpy as np
import igl
import trimesh
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from scipy.linalg import eigh


def generate_unit_sphere(num_points, radius=1.):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    sin_theta = np.sqrt(1 - cos_theta**2)


    x = sin_theta * np.cos(phi) * radius
    y = sin_theta * np.sin(phi) * radius
    z = cos_theta * radius


    points = np.column_stack((x, y, z))
    return points


def quadric(data, a, b, c):
    x, y = data
    return a * (x ** 2) + b * x * y + c * (y ** 2)


def normalize(x):
    mag = np.linalg.norm(x)
    if mag == 0:
        mag = mag + 1e-10
    return x / mag


def hat(v):
    """Generate vector right cross product matrix
    """
    if v.shape == (3, 1) or v.shape == (3,):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    else:
        raise ValueError


def local_frame(n):
    n = normalize(n)
    r = np.random.rand(3)
    t1 = normalize(np.cross(r, n))
    t2 = normalize(np.cross(n, t1))

    R = np.zeros((3, 3))
    R[:, 0] = t1
    R[:, 1] = t2
    R[:, 2] = n

    return R


class Contact():
    def __init__(self):
        self.p = None
        self.n = None
        self.t1 = None
        self.t2 = None
        self.k1 = None
        self.k2 = None
        self.grasp = None
        self.grasp_dp = None
    
    def init(self, x):
        self.p = x

    def normalize(self):
        self.n = normalize(self.n)
        self.t2 = normalize(np.cross(self.n, self.t1))
        self.t1 = normalize(np.cross(self.t2, self.n))

    def render(self):
        print('Position: ', self.p)
        print('Normal: ', self.n)
        print('Tangent 1: ', self.t1)
        print('Tangent 2: ', self.t2)
        print('Curvature 1: ', self.k1)
        print('Curvature 2: ', self.k2)
        print('Grasp: \n', contact.grasp)

    def grasp_map(self):
        R = np.zeros((3, 3))
        R[:, 0] = self.t1
        R[:, 1] = self.t2
        R[:, 2] = self.n

        grasp = np.zeros((6, 3))
        grasp[:3, :] = R
        grasp[3:, :] = hat(self.p) @ R

        R_dt1 = np.zeros((3, 3))
        R_dt1[:, 0] = self.k1 * self.n
        R_dt1[:, 2] = - self.k1 * self.t1

        grasp_dt1 = np.zeros((6, 3))
        grasp_dt1[:3, :] = R_dt1
        grasp_dt1[3:, :] = hat(self.p) @ R_dt1 + hat(self.t1) @ R
        
        R_dt2 = np.zeros((3, 3))
        R_dt2[:, 0] = self.k2 * self.n
        R_dt2[:, 2] = - self.k2 * self.t2

        grasp_dt2 = np.zeros((6, 3))
        grasp_dt2[:3, :] = R_dt2
        grasp_dt2[3:, :] = hat(self.p) @ R_dt2 + hat(self.t2) @ R

        grasp_dp = np.zeros((6, 2, 3))
        grasp_dp[:, 0, :] = grasp_dt1
        grasp_dp[:, 1, :] = grasp_dt2

        self.grasp = grasp
        self.grasp_dp = grasp_dp

        return


class PointCloud():
    def __init__(self, points, normals, k_nn=20):
        self.points = points
        self.normals = normals
        self.number = points.shape[0]
        self.k_nn = k_nn
        self.knn_model = NearestNeighbors(n_neighbors=self.k_nn, algorithm='auto', metric='euclidean')
        self.knn_model.fit(self.points)

    def sample_points(self, contact):
        contact.p = self.points[np.random.randint(self.number)]

    def kNN(self, p):
        _, indices = self.knn_model.kneighbors([p])

        knn_points = [self.points[indices[0][i]]
                       for i in range(len(indices[0]))]
        knn_points = np.array(knn_points)

        return indices[0], knn_points

    def project_geometry(self, contact):
        indice, knn = self.kNN(contact.p)

        p0 = knn[0]
        n0 = self.normals[indice[0]]
        R = local_frame(n0)

        local_knn = []
        for _p in knn:
            dp = _p - p0
            local_dp = R.transpose() @ dp
            local_knn.append(local_dp)
        local_knn = np.array(local_knn)

        (a, b, c), _ = curve_fit(quadric, (local_knn[:, 0], local_knn[:, 1]), local_knn[:, 2])

        A = np.array([[a * 2, b], 
                    [b, c * 2]])
        curv, ev = eigh(A)

        contact.p = p0
        contact.n = n0
        contact.k1 = curv[0]
        contact.k2 = curv[1]
        contact.t1 = ev[0][0] * R[:, 0] + ev[0][1] * R[:, 1]
        contact.normalize()
        return
    
    def cluster(self, local_weight=1, n_clusters=16):
        '''
        Cluster the sampled point cloud based on the points' positions and normals.

        The method clusters the sampled point cloud using KMeans clustering. The clustering is based on 
        the points' positions, their normals, and a cross product of the position and normal.

        Parameters:
        - local_weight (float, optional): Weight factor for the positions in the clustering. Default is 1.
        - n_clusters (int, optional): Number of clusters to form using KMeans clustering. Default is 16.
        '''

        t = np.empty_like(self.points) 
        for i in range(self.number):
            t[i] = normalize(np.cross(self.points[i], self.normals[i]))
        score = np.hstack([local_weight * self.points, self.normals, t]) 
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
        self.num_regions = n_clusters
        self.indices = kmeans.fit_predict(score)


class Mesh():
    def __init__(self, mesh, k_rings=3):
        self.mesh = mesh
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        
        self.number = self.vertices.shape[0]
        self.tangents1, self.tangents2, self.curvatures1, self.curvatures2 \
            = igl.principal_curvature(mesh.vertices, mesh.faces, k_rings)
        
        normals = []
        for i in range(len(self.vertices)):
            n = np.cross(self.tangents1[i], self.tangents2[i])
            n *= - np.sign(np.dot(n, mesh.vertex_normals[i]))
            normals.append(n)
        self.normals = np.vstack(normals)

    def sample_points(self, contact):
        contact.p = self.vertices[np.random.randint(self.number)]

    def project_geometry(self, contact):
        _, face_id, contact.p = igl.point_mesh_squared_distance(contact.p, self.vertices, self.faces)
        face = self.faces[face_id]

        bary_matrix = np.zeros((3, 3))
        for i in range(3):
            bary_matrix[:, i] = self.vertices[face[i]]
        bary_p = np.linalg.solve(bary_matrix, contact.p)
        bary_p /= bary_p.sum()

        contact.n = np.zeros((3,))
        contact.t1 = np.zeros((3,))
        contact.k1 = 0.0
        contact.k2 = 0.0
        for i in range(3):
            contact.n += bary_p[i] * self.normals[face[i]]
            contact.t1 += bary_p[i] * self.tangents1[face[i]] \
                * np.sign(np.dot(self.tangents1[face[i]], self.tangents1[face[0]]))
            contact.k1 += bary_p[i] * self.curvatures1[face[i]]
            contact.k2 += bary_p[i] * self.curvatures2[face[i]]
        contact.normalize()
        return







if __name__ == '__main__':
    
    # num_points = 10000
    # points = generate_unit_sphere(num_points)
    # object = PointCloud(points, - points)

    mesh = trimesh.load_mesh('./sphere.obj')
    object = Mesh(mesh)



    contact = Contact()
    # object.sample_points(contact)
    contact.p = np.array([2., 0, 0])

    object.project_geometry(contact)
    contact.grasp_map()
    contact.render()
    
