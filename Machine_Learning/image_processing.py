# Please place imports here.
# BEGIN IMPORTS
import numpy as np, scipy,cv2,imageio,nose
from scipy import ndimage
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x channels image with dimensions
                  matching the input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    height,width=images[0].shape[0:2]
    try:
        channel=images[0].shape[2]
    except:
        channel=1
    imgdat_arr = np.array(images)
    img_num = len(images)
    channels_normal=[]
    channels_albedo=[]
    for i in range(channel):
        ch=imgdat_arr[:,:,:,i].reshape(img_num,(height*width)).T
        G =  np.linalg.inv((lights).dot(lights.T)).dot((lights).dot(ch.T))
        channel_albedo=np.linalg.norm(G,axis = 0)
        channel_albedo[channel_albedo<1e-7]=0

        channel_normal=np.zeros((3,height*width))
        for i in range(height*width):
            norm=channel_albedo[i]
            channel_normal[:,i]=G[:,i]/norm if norm>=1e-7 else np.zeros(3)

        channels_albedo.append(channel_albedo)
        channels_normal.append(channel_normal)

    albedo=np.hstack(channels_albedo).reshape(channel,height*width).T.reshape(height,width,channel)
    normals=(sum(channels_normal)/len(channels_normal)).T.reshape(height,width,3)
    return albedo,normals


def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and scipy.ndimage.gaussian_filter are
    prohibited.  You must implement the separable kernel.  However, you may
    use functions such as cv2.filter2D or scipy.ndimage.correlate to do the actual
    correlation / convolution. Note that for images with one channel, cv2.filter2D
    will discard the channel dimension so add it back in.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) x channels image of type
                float32.
    """
    try:
        channel=image.shape[2]
    except:
        channel=1
    kernel=np.array([[1/16,1/4,3/8,1/4,1/16]])

    for i in range(channel):
        image[:,:,i]=ndimage.correlate(image[:,:,i],kernel,mode='mirror')
        image[:, :, i]=conv=ndimage.correlate(image[:,:,i],kernel.T,mode='mirror')

    down=image[::2,::2,:]
    return down
def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        up -- 2*height x 2*width x channels image of type float32.
    """
    height, width = image.shape[0:2]
    try:
        channel=image.shape[2]
    except:
        channel=1
    kernel=np.array([[1/8,1/2,3/4,1/2,1/8]])

    up=np.zeros([height*2,width*2,channel])
    up[::2,::2,:]=image
    up[1::2,1::2,:]=image
    for i in range(channel):
        up[:,:,i]=ndimage.correlate(up[:,:,i],kernel,mode='mirror')
        up[:, :, i]=conv=ndimage.correlate(up[:,:,i],kernel.T,mode='mirror')

    return up


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    height,width=points.shape[:2]
    mat=points.reshape((height*width,3))
    h_proj=np.dot(np.dot(K,Rt),np.vstack((mat.T,np.ones(height*width))))

    divider=np.zeros(h_proj.size).reshape(h_proj.shape)
    divider[0,:]=h_proj[2,:]
    divider[1,:]=h_proj[2,:]
    divider[2,:]=h_proj[2,:]

    h_proj=h_proj/divider

    projections=h_proj[:2,:].T.reshape((height,width,2))
    return projections

def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R' -R't |
            | 0 1 |      | 0   1   |

          p = R' * (z * x', z * y', z, 1)' - R't

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """

    DDPoints=np.array([[0,0,width,0,0,height,width,height]]).reshape((4,2))
    h_DDPoints=np.vstack((DDPoints.T,np.ones(4)))
    loc_direct=np.dot(np.linalg.inv(K),h_DDPoints)

    divider=np.zeros(h_DDPoints.size).reshape(h_DDPoints.shape)
    divider[0,:]=h_DDPoints[2,:]
    divider[1,:]=h_DDPoints[2,:]
    divider[2,:]=h_DDPoints[2,:]

    unit_depth_loc_direct=loc_direct/divider
    z_depth_loc_direct=depth*unit_depth_loc_direct
    Rinv_t=np.dot(np.linalg.inv(Rt[:,:3]),Rt[:,3])
    points_mat=np.dot(np.linalg.inv(Rt[:,:3]),z_depth_loc_direct)-np.hstack((Rinv_t,Rinv_t,Rinv_t,Rinv_t)).reshape(4,3).T
    points=points_mat.T.reshape((2,2,3))
    return points

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height,width=image.shape[0:2]
    non_normalized_dummy=np.zeros(height*width)
    try:
        channel=image.shape[2]
    except:
        channel=1
    for i in range(channel):
        patches_Mean=np.zeros((height,width,ncc_size,ncc_size))
        for j in range(height-ncc_size+1):
            for k in range(width-ncc_size+1):
                patches_Mean[j+ncc_size//2,k+ncc_size//2]=image[:,:,i][j:j+ncc_size,k:k+ncc_size]-np.mean(image[:,:,i][j:j+ncc_size,k:k+ncc_size])
        non_normalized_dummy=np.vstack((non_normalized_dummy,patches_Mean.reshape((height*width,ncc_size*ncc_size)).T))
    non_normalized=non_normalized_dummy[1:,:]
    normalized=np.zeros((ncc_size*ncc_size*channel,height*width))
    for i in range(height*width):
        norm=np.linalg.norm(non_normalized[:,i])
        normalized[:,i]=non_normalized[:,i]/norm if norm>=0.000001 else np.zeros(ncc_size*ncc_size*channel)

    normalized=normalized.T.reshape((height,width,ncc_size*ncc_size*channel))
    return normalized
def compute_ncc_impl(image1, image2):

    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height,width,arr=image1.shape[0:3]
    ncc=np.sum((image1.flatten()*image2.flatten()).reshape(height*width,arr).T,axis=0).reshape(height,width)
    return ncc
