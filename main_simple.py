import numpy as np
import cv2
import math
from objloader_simple import *

# class ObjLoader(object):
#     def __init__(self, fileName):
#         self.vertices = []
#         self.faces = []
#         ##
#         try:
#             f = open(fileName)
#             for line in f:
#                 if line[:2] == "v ":
#                     index1 = line.find(" ") + 1
#                     index2 = line.find(" ", index1 + 1)
#                     index3 = line.find(" ", index2 + 1)

#                     vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
#                     vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
#                     self.vertices.append(vertex)

#                 elif line[0] == "f":
#                     string = line.replace("//", "/")
#                     ##
#                     i = string.find(" ") + 1
#                     face = []
#                     for item in range(string.count(" ")):
#                         if string.find(" ", i) == -1:
# 							print (string[i:-1])
#                             face.append(string[i:-1])
#                             break
#                         face.append(string[i:string.find(" ", i)])
#                         i = string.find(" ", i) + 1
#                     ##
#                     self.faces.append(tuple(face))

#             f.close()
#         except IOError:
#             print(".obj file not found.")

def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.5
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
# """
#  From the camera calibration matrix and the estimated homography
#  compute the 3D projection matrix
#  """
# Compute rotation along the x and y axis as well as the translation
	homography = homography * (-1)
	rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]
	# normalise vectors
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	translation = col_3 / l
	# compute the orthonormal basis
	c = rot_1 + rot_2
	p = np.cross(rot_1, rot_2)
	d = np.cross(c, p)
	rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_3 = np.cross(rot_1, rot_2)
	# finally, compute the 3D projection matrix from the model to the current frame
	projection = np.stack((rot_1, rot_2, rot_3, translation)).T
	return np.dot(camera_parameters, projection)

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
	(hA, wA) = imageA.shape[:2]
	(hB, wB) = imageB.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB), dtype="uint8")
	vis[0:hA, 0:wA] = imageA
	vis[0:hB, wA:] = imageB

	for ((trainIdx, queryIdx), s) in zip(matches, status):
	    # only process the match if the keypoint was successfully
	    # matched
	    if s == 1:
	        ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
	        ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
	        cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

	return vis


MIN_MATCHES = 25
ratio = 0.75
reprojThresh = 4
camera_parameters = np.array([[601.1,0,332.86],[0,600.21,226.4],[0,0,1]])
# camera_parameters[0,0] = 985.458
# camera_parameters[1,1] = 1015.459
# camera_parameters[0,2] = 501.0931
# camera_parameters[1,2] = 227.777
# camera_parameters[2,2] = 1
print "camera_parameters: ",camera_parameters

obj = OBJ('cow.obj', swapyz=True)
# cap = cv2.imread('scene1.jpg', 0)    
model = cv2.imread('modelA.jpg', 0)
# ORB keypoint detector
orb = cv2.xfeatures2d.SIFT_create()
# orb = cv2.ORB_create()       

kp_model, des_model = orb.detectAndCompute(model, None)  
kp_m = np.float32([kp.pt for kp in kp_model])
matcher = cv2.DescriptorMatcher_create("BruteForce")
cap = cv2.VideoCapture(2)

ret = False
while True:
	# read the current frame
	ret, frame = cap.read()
	print not ret
	cv2.imshow('frame', frame)
	if (not ret):
	    print "Unable to capture video"
	    break

	print ("here")
	kp_frame, des_frame = orb.detectAndCompute(frame, None)
	kp_f = np.float32([kp.pt for kp in kp_frame])
	matches = []
	if des_frame is not None:
		rawMatches = matcher.knnMatch(des_model, des_frame, 2)
		print "here"

		for m in rawMatches:
		    # ensure the distance is within a certain ratio of each
		    # other (i.e. Lowe's ratio test)
		    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		        matches.append((m[0].trainIdx, m[0].queryIdx))

		print (len(matches))
		if len(matches) > MIN_MATCHES:
			try: 
				ptsA = np.float32([kp_m[i] for (_, i) in matches])
				ptsB = np.float32([kp_f[i] for (i, _) in matches])

				# compute the homography between the two sets of points
				(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				    reprojThresh)

				# cap1 = drawMatches(model, frame, kp_m, kp_f, matches, status)
				# cv2.imshow('frame', cap1)
				# cv2.waitKey(0)

				# Draw a rectangle that marks the found model in the frame
				h, w = model.shape
				pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
				# project corners into frame
				dst = cv2.perspectiveTransform(pts, H)  
				# connect them with lines
				img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
				# cv2.imshow('frame', frame)
				# cv2.waitKey(0)

				# if a valid homography matrix was found render cube on model plane
				if H is not None:
					try:
						# H1 = np.zeros(shape = (3,4))
						# H1[0][0] = 1
						# H1[1][1] = 1
						# H1[2][2] = 1
						# P = projection_matrix(camera_parameters, H1)  
						# project cube or model
						# frame = render(model, obj, P, model, False)
						# cv2.imshow('frame', frame)
						# cv2.waitKey(0)

						# obtain 3D projection matrix from homography matrix and camera parameters
						projection = projection_matrix(camera_parameters, H)  
						print projection
						
						# project cube or model
						frame = render(frame, obj, projection, model, False)
						# cv2.imshow('frame', frame)
						# cv2.waitKey(0)
					except Exception as e:
						print (e)
						print ("except")
						pass
			except Exception as e:
				print (e)
				print ("except 1")
				pass
			
		else:
		    print ("Not enough matches have been found - %d/%d" % (len(matches),
		                                                          MIN_MATCHES))
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
cap.release()
cv2.destroyAllWindows()