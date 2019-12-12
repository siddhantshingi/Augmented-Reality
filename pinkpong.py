import numpy as np
import cv2
import math
import random
from objloader_simple import *

def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.01
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
thres = 50
rate = 100
speed = 1.1
# acquired by calibration of camera
camera_parameters = np.array([[601.1,0,332.86],[0,600.21,226.4],[0,0,1]])

obj = OBJ('object_files/pokeball.obj', swapyz=True)
modelA = cv2.imread('markers/modelB.jpg', 0)
modelB = cv2.imread('markers/modelB_p.jpg', 0)

orb = cv2.xfeatures2d.SIFT_create()
kp_modelA, des_modelA = orb.detectAndCompute(modelA, None)  
kp_modelB, des_modelB = orb.detectAndCompute(modelB, None)  
matcher = cv2.DescriptorMatcher_create("BruteForce")
kp_mA = np.float32([kp.pt for kp in kp_modelA])
kp_mB = np.float32([kp.pt for kp in kp_modelB])

cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('examples/ping_pong_game.mp4', fourcc, fps, (width,height))

intersect = False
start = True
ret = False
render_obj = False
count = 0
player = 0
while True:
	detect_A = False
	detect_B = False
	ret, frame = cap.read()
	cv2.imshow('frame', frame)
	if (not ret):
	    print "Unable to capture video"
	    break
	cv2.line(frame,(int(frame.shape[1]/4),0),(int(frame.shape[1]/4),frame.shape[0]),(255,255,255),5)
	cv2.line(frame,(3*int(frame.shape[1]/4),0),(3*int(frame.shape[1]/4),frame.shape[0]),(255,255,255),5)

	kp_frame, des_frame = orb.detectAndCompute(frame, None)
	kp_f = np.float32([kp.pt for kp in kp_frame])
	if des_frame is not None:
		rawMatches = matcher.knnMatch(des_modelA, des_frame, 2)
		matches = []

		for m in rawMatches:
		    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		        matches.append((m[0].trainIdx, m[0].queryIdx))

		if len(matches) > MIN_MATCHES:
			try: 
				ptsA = np.float32([kp_mA[i] for (_, i) in matches])
				ptsB = np.float32([kp_f[i] for (i, _) in matches])
				(H_A, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				    reprojThresh)

				h, w = modelA.shape
				pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
				dst_A = cv2.perspectiveTransform(pts, H_A)  

				if H_A is not None:
					try:
						(ext_A, projection_A) = projection_matrix(camera_parameters, H_A)  
						mid_pt = np.array([(w - 1)/2, (h -1)/2, 0, 1])
						dst_mid_pt = np.dot(projection_A,mid_pt)
						x_A = int(dst_mid_pt[0]/dst_mid_pt[2]) 
						y_A = int(dst_mid_pt[1]/dst_mid_pt[2]) 
						centerA=(x_A,y_A)
						radiusA=(w+h)/32
						frame = cv2.circle(frame,centerA,radiusA,(255,0,0),3)
						frame = cv2.circle(frame,centerA,3,(255,0,0),5)
						detect_A = True
					except Exception as e:
						pass
			except Exception as e:
				pass
			
		rawMatches = matcher.knnMatch(des_modelB, des_frame, 2)
		matches = []

		for m in rawMatches:
		    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		        matches.append((m[0].trainIdx, m[0].queryIdx))

		if len(matches) > MIN_MATCHES:
			try: 
				ptsA = np.float32([kp_mB[i] for (_, i) in matches])
				ptsB = np.float32([kp_f[i] for (i, _) in matches])

				(H_B, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				    reprojThresh)

				h, w = modelB.shape
				pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
				dst_B = cv2.perspectiveTransform(pts, H_B)  

				if H_B is not None:
					try:
						(ext_B, projection_B) = projection_matrix(camera_parameters, H_B)  
						mid_pt = np.array([(w - 1)/2, (h -1)/2, 0, 1])
						dst_mid_pt = np.dot(projection_B,mid_pt)
						x_B = int(dst_mid_pt[0]/dst_mid_pt[2]) 
						y_B = int(dst_mid_pt[1]/dst_mid_pt[2]) 
						centerB=(x_B,y_B)
						radiusB=(w+h)/32
						frame = cv2.circle(frame,centerB,radiusB,(0,0,255),3)
						frame = cv2.circle(frame,centerB,3,(0,0,255),5)
						detect_B = True
					except Exception as e:
						pass
			except Exception as e:
				pass
			
		if detect_A and detect_B and start:
			count += 1
			if (count > thres):
				start = False
				rx_A = random.randint(int(frame.shape[1]/4),3*int(frame.shape[1]/8))
				rx_B = random.randint(5*int(frame.shape[1]/8),3*int(frame.shape[1]/4))
				ry_A = random.randint(0,frame.shape[0])
				ry_B = random.randint(0,frame.shape[0])
				c_A = (rx_A,ry_A)
				c_B = (rx_B,ry_B)
				m1 = (ry_B - ry_A)*1.0/(rx_B - rx_A)
				d = math.sqrt((ry_B - ry_A)*(ry_B - ry_A) + (rx_B - rx_A)*(rx_B - rx_A))
				render_obj = True

		if intersect:
			intersect = False
			rate = rate/speed
			m3 = (m1*m2*m2 + 2*m2 - m1)*1.0/(1 + 2*m1*m2 - m2*m2)
			x1 = c[0] + d*1.0/math.sqrt(1+m3*m3)
			y1 = c[1] + m3*d*1.0/math.sqrt(1+m3*m3)
			x2 = c[0] - d*1.0/math.sqrt(1+m3*m3)
			y2 = c[1] - m3*d*1.0/math.sqrt(1+m3*m3)
			vec1 = (x1 - c[0], y1 - c[1])
			if (player == -1):
				nor_vec = (c[0] - x_A, c[1] - y_A)
			else:
				nor_vec = (c[0] - x_B, c[1] - y_B)
			dot1 = vec1[0]*nor_vec[0] + vec1[1]*nor_vec[1]
			if (dot1 >= 0):
				x = x1
				y = y1
			else:
				x = x2
				y = y2
			count = thres
			c_A = (int(c[0]),int(c[1]))
			c_B = (int(x),int(y))
			m1 = m3

		if render_obj:
			Lambda = (count - thres)*1.0/rate
			c = (c_A[0] + Lambda*(c_B[0] - c_A[0]),c_A[1] + Lambda*(c_B[1] - c_A[1]))
			if (c[0]<0 or c[0]>frame.shape[1]):
				game_over = True
				print ("GAME OVER")
				if (player == 0):
					print ("RED PLAYER WON")
				elif (player == -1):
					print ("BLUE PLAYER WON")
				elif (player == 1):
					print ("RED PLAYER WON")
				break
			if (c[1]<=0):
				m3 = -1.0*m1
				if (m3 < 0):
					x = c[0] - d*1.0/math.sqrt(1+m3*m3)
					y = -1.0*m3*d/math.sqrt(1+m3*m3)
				else:
					x = c[0] + d*1.0/math.sqrt(1+m3*m3)
					y = 1.0*m3*d/math.sqrt(1+m3*m3)
				count = thres
				c_A = (int(c[0]),0)
				c_B = (int(x),int(y))
				m1 = m3
			if (c[1]>=frame.shape[0]):
				m3 = -1.0*m1
				if (m3 < 0):
					x = c[0] + d*1.0/math.sqrt(1+m3*m3)
					y = frame.shape[0] + m3*d*1.0/math.sqrt(1+m3*m3)
				else:
					x = c[0] - d*1.0/math.sqrt(1+m3*m3)
					y = frame.shape[0] - m3*d*1.0/math.sqrt(1+m3*m3)
				count = thres
				c_A = (int(c[0]),frame.shape[0])
				c_B = (int(x),int(y))
				m1 = m3
			dist_A = math.sqrt((x_A - c[0])*(x_A - c[0]) + (y_A - c[1])*(y_A - c[1]))
			dist_B = math.sqrt((x_B - c[0])*(x_B - c[0]) + (y_B - c[1])*(y_B - c[1]))
			if (dist_A <= radiusA and player != -1):
				m2 = (c[1] - y_A)*1.0/(c[0] - x_A)
				player = -1
				intersect = True
			elif (dist_B <= radiusB and player != 1):
				player = 1
				m2 = (c[1] - y_B)*1.0/(c[0] - x_B)
				intersect = True
			frame = cv2.circle(frame,(int(c[0]),int(c[1])),8,(0,255,0),8)
			count += 1
	cv2.imshow('frame', frame)
	out.write(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
cap.release()
out.release()
cv2.destroyAllWindows()