import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as msg
import configparser as cp
import ntpath
from PIL import Image, ImageTk
from PIL.ExifTags import TAGS
import cv2 as cv
import numpy as np
import sys, os
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

class IniEditor(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Peacock feather ocelli detection")
        self.geometry("1200x900")

        self.active_ini = ""
        self.active_ini_filename = ""
        self.ini_elements = {}
        self.left_frame = tk.Frame(self, width=800, bg="grey")
        self.left_frame.pack_propagate(0)

        self.right_frame = tk.Frame(self, width=400, bg="lightgrey")
        self.right_frame.pack_propagate(0)

        self.METHOD_TYPE = tk.StringVar(self)
        self.METHOD_TYPE.set("Detection Using Template Matching (Sift)")

        self.method_label = tk.OptionMenu(self, self.METHOD_TYPE, "Detection Using Template Matching (Sift)", "Detection Using Hough Transform", command=self.update_method)
        self.method_label.pack(side=tk.TOP, expand=1, fill=tk.X, anchor="n")

        self.select_template = False
        self.started_selecting = False

        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        self.right_frame.pack(side=tk.LEFT, expand=1, fill=tk.BOTH)

        self.right_frame.bind("<Configure>", self.frame_height)


        self.template = ImageTk.PhotoImage(Image.open("feather1.jpg").resize((100, 100),Image.ANTIALIAS))
        self.t_img = Image.open("feather1.jpg")

        self.message = None
        self.rectangle = None
        self.canvas_image = None
        self.template_canvas_image = None
        self.canvas_message = None
        self.files = []
        self.box = [0, 0, 0, 0]
        self.ratio = 1.0
        self.canvas = tk.Canvas(self.left_frame,
                                    highlightthickness=0,
                                    bd=0)
        self.template_canvas = tk.Canvas(self.right_frame,
                                    highlightthickness=0,
                                    bd=0)

        self.bind("<Button-1>", self.__on_mouse_down)
        self.bind("<ButtonRelease-1>", self.__on_mouse_release)
        self.bind("<B1-Motion>", self.__on_mouse_move)

        # self.bind("<Control-n>", self.file_new)
        self.bind("<Control-o>", self.file_open)
        self.bind('<Return>', self.detect_ocelli)
        # self.bind("<Control-s>", self.file_save)


        self.MATCH_METHOD = tk.IntVar()
        self.MATCH_METHOD.set(1)
        self.MATCH_TH = tk.IntVar()
        self.MATCH_TH.set(50)
        self.MATCH_OCELLI = tk.StringVar(self)
        self.MATCH_OCELLI.set("0")
        self.IMAGE_TYPE = tk.IntVar()
        self.IMAGE_TYPE.set(1)
        self.file_open()

    def update_method(self,e=None):
        self.IMAGE_TYPE.set(1)
        self.detect_ocelli()
        self.display_section_contents()
    
    def frame_height(self, event=None):
        new_height = self.winfo_height()
        self.right_frame.configure(height=new_height)

    def file_open(self, event=None):
        ini_file = filedialog.askopenfilename(filetypes=[('Image files', ('.png', '.jpg', '.jpeg'))])
        if ini_file:
            self.set_image(ini_file)
            self.display_section_contents()
            self.detect_ocelli()


    def clear_right_frame(self):
        for child in self.right_frame.winfo_children():
            child.destroy()

    def pil2bgr(self,img):
        img_rgb = np.array(img)
        return img_rgb[:, :, ::-1].copy() 
        

    def detect_ocelli_template_matching(self,e=None):
        img_rgb = self.pil2bgr(self.img)
        img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        template = self.pil2bgr(self.t_img)
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]
        methods = {
            1:cv.TM_CCOEFF_NORMED,
            2:cv.TM_CCORR_NORMED,
            3:cv.TM_SQDIFF_NORMED,
        }
        res = cv.matchTemplate(img_gray,template,methods[self.MATCH_METHOD.get()])
        loc = np.where( res >= self.MATCH_TH.get()/100)
        img_new = np.zeros_like(img_rgb)
        self.bare_img_other = np.copy(img_rgb)
        for pt in zip(*loc[::-1]):
            cv.circle(img_new,pt,3,(255,255,255),-1)
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
        img_new = cv.cvtColor(img_new, cv.COLOR_BGR2GRAY)
        self.bare_img = Image.fromarray(img_new)
        conn_comp = cv.connectedComponentsWithStats(img_new)
        stats = conn_comp[2]
        for i in range(conn_comp[0]):
            if stats[i,cv.CC_STAT_WIDTH]>img_rgb.shape[1]*2//3 or stats[i,cv.CC_STAT_HEIGHT]>img_rgb.shape[0]*2//3:
                continue
            cv.rectangle(self.bare_img_other, (stats[i,cv.CC_STAT_LEFT],stats[i,cv.CC_STAT_TOP]) , (stats[i,cv.CC_STAT_LEFT]+w,stats[i,cv.CC_STAT_TOP]+h), (0,0,255), 1)
        self.bare_img_other = Image.fromarray(self.bare_img_other)
        self.MATCH_OCELLI.set(str(int(conn_comp[0])-1))
        return Image.fromarray(img_rgb)

    def detect_ocelli_hough_transform(self,e=None):
        img_rgb = self.pil2bgr(self.img)
        img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

        high_thresh, thresh_im = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if self.OTSU_THRES.get()==-1:
            self.OTSU_THRES.set(high_thresh)
            self.MATCH_MAX_TH.set(high_thresh)
            self.MATCH_MIN_TH.set(high_thresh//2)

        self.bare_img = img_as_ubyte(img_gray)
        self.bare_img = canny(self.bare_img, sigma=float(self.SIGMA.get()), low_threshold=self.MATCH_MIN_TH.get(), high_threshold=self.MATCH_MAX_TH.get())

        hough_radii = np.arange(self.MIN_RAD.get(), self.MAX_RAD.get(), 1)
        hough_res = hough_circle(self.bare_img, hough_radii)

        accums, cx, cy, radii = hough_circle_peaks(
            hough_res, hough_radii, min_xdistance=self.MIN_DIST.get(), min_ydistance=self.MIN_DIST.get())

        self.MATCH_OCELLI.set(cx.shape[0])

        self.bare_img_other = Image.fromarray(self.bare_img)
        self.bare_img = cv.cvtColor(self.bare_img.astype('uint8')*255,cv.COLOR_GRAY2RGB)
        # Draw them on the image
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=img_rgb.shape)
            img_rgb[circy, circx] = (0, 0, 255)
            self.bare_img[circy, circx] = (0, 0, 255)
        self.bare_img = Image.fromarray(self.bare_img)
        return Image.fromarray(img_rgb)

    def detect_ocelli(self,e=None):
        val = self.METHOD_TYPE.get()
        img = None
        if val == "Detection Using Template Matching (Sift)":
            img = self.detect_ocelli_template_matching()
        else:
            img = self.detect_ocelli_hough_transform()
        val = self.IMAGE_TYPE.get()
        if val!=1:
            self.changeimage()
        else:
            self.set_image(None,img)
    
    def changeimage(self):
        val = self.IMAGE_TYPE.get()
        if val==1:
            self.detect_ocelli()
        elif val==2:
            self.set_image(None,self.img)
        elif val==3:
            self.set_image(None,self.bare_img)
        else:
            self.set_image(None,self.bare_img_other)


if __name__ == "__main__":
    ini_editor = IniEditor()
    ini_editor.mainloop()
