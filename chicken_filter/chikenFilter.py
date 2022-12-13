import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

img_s = [224, 168]

def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA

    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h
    saspect = float(sw)/sh

    if (saspect >= aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(
            int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(
            int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    # color image but only one color provided
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def loadInterpreter():
    interpreter = tflite.Interpreter(model_path='chicken_filter/model.tflite')
    return interpreter


def predictAI(img, interpreter):
    result = -1
    imgage = resizeAndPad(img, (img_s[1], img_s[0]), 0)
    imgage = imgage.astype(np.float32)
    input_data = imgage[None, :]

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if (np.argmax(output_data)) == 0:
        # Bad Images
        if np.max(output_data) > 0.60:
            return 0
        else:
            return 0
    else:
        # Good Images
        if np.max(output_data) > 0.85:
            return 1
        else:
            return 1

def imgRead(img, interpreter):
    if (np.mean(img) > 10 and np.mean(img) < 180):
        return predictAI(img, interpreter)
    else:
        return 0

def readDir(interpreter, dirPath, outdir):
    count = 0
    try:
        os.makedirs(outdir+'0/')
        os.makedirs(outdir+'1/')
        os.makedirs(outdir+'2/')
    except Exception as e:
        print(e)
    for i in os.listdir(dirPath):
        imgPath = dirPath+i
        img = cv2.imread(imgPath)
        res = imgRead(img, interpreter)
        cv2.imwrite(outdir+str(res)+'/image-'+str(count)+'.jpg', img)
        count += 1

if __name__ == "__main__":
    inte = loadInterpreter('model.tflite')
    outDir = 'output/'
    dataDir = 'data/All/'
    readDir(inte, dataDir, outDir)
