import cv2
import numpy as np
import dlib

def transFace(img):
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    #人脸分类器
    detector = dlib.get_frontal_face_detector()
    # 获取人脸检测器
    predictor = dlib.shape_predictor("/home/xuwanqian/Download/shape_predictor_68_face_landmarks.dat")

    dets = detector(gray, 1)
    for face in dets:
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        ROIp = shape.parts()[:27]
        pts = []
        shape = gray.shape
        rows, cols = shape
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in ROIp:
            pt_pos = (pt.x, pt.y)
            pts.append(pt_pos)
            #  cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
        ptsRightEye = pts[19]
        ptsLeftEye = pts[24]
        center = pts[8]
        faceWidth = pts[16][0]-pts[0][0]
        rightEyep = cols//2-(cols/faceWidth*abs(cols//2-ptsRightEye[0]))
        leftEyep = cols//2+(cols/faceWidth*abs(cols//2-ptsLeftEye[0]))
        transRightEye = [rightEyep, 0]
        transLeftEye = [leftEyep, 0]
        transCenter = [cols//2, rows]
        pts1, pts2 = pts[:17], pts[17:]
        pts = pts1 + pts2[::-1]
        mask = np.zeros(shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
        ROI = cv2.bitwise_and(mask2, gray)

        pts1 = np.float32([ptsLeftEye, ptsRightEye, center])
        pts2 = np.float32([transLeftEye, transRightEye, transCenter])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(ROI, M, (cols, rows))
        #  cv2.imshow("ROI", dst)
        #  cv2.imwrite(saveName, dst)
        return dst

if __name__=="__main__":
    img = cv2.imread("./test.jpg")
    img = transFace(img)
    cv2.imshow("1", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
