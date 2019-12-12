import numpy as np
import cv2
import math
from objloader_simple import *

def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.5
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
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
	homography = homography * (-1)
	rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	translation = col_3 / l
	c = rot_1 + rot_2
	p = np.cross(rot_1, rot_2)
	d = np.cross(c, p)
	rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_3 = np.cross(rot_1, rot_2)
	projection = np.stack((rot_1, rot_2, rot_3, translation)).T
	return (projection,np.dot(camera_parameters, projection))

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
	(hA, wA) = imageA.shape[:2]
	(hB, wB) = imageB.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB), dtype="uint8")
	vis[0:hA, 0:wA] = imageA
	vis[0:hB, wA:] = imageB

	for ((trainIdx, queryIdx), s) in zip(matches, status):
	    if s == 1:
	        ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
	        ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
	        cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

	return vis


MIN_MATCHES = 25
ratio = 0.75
reprojThresh = 4
# acquired by calibration of camera
camera_parameters = np.array([[601.1,0,332.86],[0,600.21,226.4],[0,0,1]])

obj = OBJ('object_files/cow.obj', swapyz=True)
model = cv2.imread('markers/model1.jpg', 0)
orb = cv2.xfeatures2d.SIFT_create()

kp_model, des_model = orb.detectAndCompute(model, None)  
kp_m = np.float32([kp.pt for kp in kp_model])
matcher = cv2.DescriptorMatcher_create("BruteForce")

cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('examples/simple_AR.mp4', fourcc, fps, (width,height))


ret = False
while True:
	ret, frame = cap.read()
	cv2.imshow('frame', frame)
	if (not ret):
	    print "Unable to capture video"
	    break

	kp_frame, des_frame = orb.detectAndCompute(frame, None)
	kp_f = np.float32([kp.pt for kp in kp_frame])
	matches = []
	if des_frame is not None:
		rawMatches = matcher.knnMatch(des_model, des_frame, 2)

		for m in rawMatches:
		    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		        matches.append((m[0].trainIdx, m[0].queryIdx))

		if len(matches) > MIN_MATCHES:
			try: 
				ptsA = np.float32([kp_m[i] for (_, i) in matches])
				ptsB = np.float32([kp_f[i] for (i, _) in matches])

				(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				    reprojThresh)

				h, w = model.shape
				pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
				dst = cv2.perspectiveTransform(pts, H)  
				img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
				if H is not None:
					try:
						(ext, projection) = projection_matrix(camera_parameters, H)  
						frame = render(frame, obj, projection, model, False)
					except Exception as e:
						pass
			except Exception as e:
				pass
			
	cv2.imshow('frame', frame)
	out.write(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
cap.release()
out.release()
cv2.destroyAllWindows()