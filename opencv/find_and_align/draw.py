import numpy as np
import cv2

def draw_match_rect(image, corners, color=(255,255,255)):
	cv2.polylines(image, corners, True, color)


def draw_match_line(image, x1, x2, y1, y2, r, color=(0,255,0)):
	cv2.circle(image, (x1, y1), r, color, -1)
	cv2.circle(image, (x2, y2), r, color, -1)
	cv2.line(image, (x1, y1), (x2, y2), color)


def draw_cross(image, x1, x2, y1, y2, r, color=(0,0,255), thickness=3):
	cv2.line(image, (x1-r, y1-r), (x1+r, y1+r), color, thickness)
	cv2.line(image, (x1-r, y1+r), (x1+r, y1-r), color, thickness)
	cv2.line(image, (x2-r, y2-r), (x2+r, y2+r), color, thickness)
	cv2.line(image, (x2-r, y2+r), (x2+r, y2-r), color, thickness)
