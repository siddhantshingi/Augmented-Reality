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
    scale_matrix = np.eye(3) * 0.01
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
	return (projection,np.dot(camera_parameters, projection))

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
thres = 100
rate = 100
camera_parameters = np.array([[601.1,0,332.86],[0,600.21,226.4],[0,0,1]])
# camera_parameters[0,0] = 985.458
# camera_parameters[1,1] = 1015.459
# camera_parameters[0,2] = 501.0931
# camera_parameters[1,2] = 227.777
# camera_parameters[2,2] = 1
print "camera_parameters: ",camera_parameters

obj = OBJ('pokeball.obj', swapyz=True)
modelA = cv2.imread('modelB.jpg', 0)
modelB = cv2.imread('modelA.jpg', 0)

orb = cv2.xfeatures2d.SIFT_create()
kp_modelA, des_modelA = orb.detectAndCompute(modelA, None)  
kp_modelB, des_modelB = orb.detectAndCompute(modelB, None)  
matcher = cv2.DescriptorMatcher_create("BruteForce")
kp_mA = np.float32([kp.pt for kp in kp_modelA])
kp_mB = np.float32([kp.pt for kp in kp_modelB])

cap = cv2.VideoCapture(2)

detect_A = False
detect_B = False
start = False
ret = False
count=0
while True:
	ret, frame = cap.read()
	cv2.imshow('frame', frame)
	if (not ret):
	    print "Unable to capture video"
	    break

	if count >= thres:
		# print projection_A, projection_B
		# projection = projection_A
		Lambda = (count - thres)*1.0/rate
		if (Lambda <= 1): 
			cv2.line(frame,(x_A,y_A),(x_B,y_B),(0,255,0),5)
		# 	print Lambda	
		# 	R1 = ext_A[:,0:3]
		# 	R2 = ext_B[:,0:3]
		# 	print R1.shape
		# 	L1 = ext_A[:,3:]
		# 	L2 = ext_B[:,3:]
		# 	projection[:,3:] = (1 + Lambda)*L1 - Lambda*np.dot(R1,np.dot(np.linalg.inv(R2),L2))
			translate=[[1,0,Lambda*(x_B - x_A)],[0,1,Lambda*(y_B - y_A)],[0,0,1]]
			projection = np.dot(translate,projection_A)
		frame = render(frame, obj, projection, modelB, False)
		cv2.imshow('frame', frame)
		count = count + 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break
		continue

	kp_frame, des_frame = orb.detectAndCompute(frame, None)
	kp_f = np.float32([kp.pt for kp in kp_frame])

	if des_frame is not None:
		rawMatches = matcher.knnMatch(des_modelA, des_frame, 2)
		matches = []

		for m in rawMatches:
		    # ensure the distance is within a certain ratio of each
		    # other (i.e. Lowe's ratio test)
		    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		        matches.append((m[0].trainIdx, m[0].queryIdx))

		# print (len(matches))
		if len(matches) > MIN_MATCHES:
			try: 
				ptsA = np.float32([kp_mA[i] for (_, i) in matches])
				ptsB = np.float32([kp_f[i] for (i, _) in matches])

				# compute the homography between the two sets of points
				(H_A, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				    reprojThresh)

				# cap1 = drawMatches(modelA, frame, kp_mA, kp_f, matches, status)
				# cv2.imshow('frame', cap1)
				# cv2.waitKey(0)

				# Draw a rectangle that marks the found modelA in the frame
				h, w = modelA.shape
				pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
				# project corners into frame
				dst_A = cv2.perspectiveTransform(pts, H_A)  
				# connect them with lines
				# img2 = cv2.polylines(frame, [np.int32(dst_A)], True, 255, 3, cv2.LINE_AA) 

				# if a valid homography matrix was found render cube on modelA plane
				if H_A is not None:
					try:
						# obtain 3D projection matrix from homography matrix and camera parameters
						(ext_A, projection_A) = projection_matrix(camera_parameters, H_A)  
						mid_pt = np.array([(w - 1)/2, (h -1)/2, 0, 1])
						dst_mid_pt = np.dot(projection_A,mid_pt)
						x_A = int(dst_mid_pt[0]/dst_mid_pt[2]) 
						y_A = int(dst_mid_pt[1]/dst_mid_pt[2]) 
						frame = cv2.circle(frame,(x_A,y_A),3,(0,0,255),5)
						detect_A = True
						# print projection_A						
						# project cube or modelA
						# frame = render(frame, obj, projection, modelA, False)
					except Exception as e:
						print (e)
						print ("except")
						pass
			except Exception as e:
				print (e)
				print ("except 1")
				pass
			
		else:
		    print ("Not enough matches have been found for A- %d/%d" % (len(matches),
		                                                          MIN_MATCHES))

		rawMatches = matcher.knnMatch(des_modelB, des_frame, 2)
		matches = []

		for m in rawMatches:
		    # ensure the distance is within a certain ratio of each
		    # other (i.e. Lowe's ratio test)
		    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		        matches.append((m[0].trainIdx, m[0].queryIdx))

		# print (len(matches))
		if len(matches) > MIN_MATCHES:
			try: 
				ptsA = np.float32([kp_mB[i] for (_, i) in matches])
				ptsB = np.float32([kp_f[i] for (i, _) in matches])

				# compute the homography between the two sets of points
				(H_B, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				    reprojThresh)

				# cap1 = drawMatches(modelA, frame, kp_mA, kp_f, matches, status)
				# cv2.imshow('frame', cap1)
				# cv2.waitKey(0)

				# Draw a rectangle that marks the found modelA in the frame
				h, w = modelB.shape
				pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
				# project corners into frame
				dst_B = cv2.perspectiveTransform(pts, H_B)  
				# connect them with lines
				# img2 = cv2.polylines(frame, [np.int32(dst_B)], True, (0,0,255), 3, cv2.LINE_AA) 

				# if a valid homography matrix was found render cube on modelA plane
				if H_B is not None:
					try:
						# obtain 3D projection matrix from homography matrix and camera parameters
						(ext_B, projection_B) = projection_matrix(camera_parameters, H_B)  
						mid_pt = np.array([(w - 1)/2, (h -1)/2, 0, 1])
						dst_mid_pt = np.dot(projection_B,mid_pt)
						x_B = int(dst_mid_pt[0]/dst_mid_pt[2]) 
						y_B = int(dst_mid_pt[1]/dst_mid_pt[2]) 
						frame = cv2.circle(frame,(x_B,y_B),3,(0,0,255),5)
						detect_B = True
						# print projection_A						
						# project cube or modelA
						# frame = render(frame, obj, projection, modelA, False)
					except Exception as e:
						print (e)
						print ("except")
						pass
			except Exception as e:
				print (e)
				print ("except 1")
				pass
			
		else:
		    print ("Not enough matches have been found for B - %d/%d" % (len(matches),
		                                                          MIN_MATCHES))
		if (detect_A and detect_B) or start:
			start = True
			cv2.line(frame,(x_A,y_A),(x_B,y_B),(0,255,0),5)
			count=count+1
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
cap.release()
cv2.destroyAllWindows()