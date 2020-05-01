from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyzbar.pyzbar import decode

class QrBase():
    """
    QRコードの関数をまとめたクラス
    Staticmethodを利用

    """
    
    @staticmethod
    def qr_reader(image_file, posetime=2, image_save=False):
        """

        # 画像ファイルの指定
        image_file = './photos/QR_Test.jpg'
        """

        # QRコードの読取り
        img_qr = cv2.imread(str(image_file))
        qr_data = decode(img_qr)
        # print(qr_data)
        # plt.imshow(img_qr)

        if qr_data != []:
            for  symbol in qr_data:
                # symbolの内容
                # [0][0]->.data, [0][1]->.type, [0][2]->rect, [0][3]->polygon
                # example
                # for obj in decoded_objs:
                #     print('Type: ', obj.type)

                str_data =symbol.data.decode('utf-8', 'ignore')
                print('QR cord: {}'.format(str_data))
                left, top, width, height = symbol.rect
                centerx = int(left+(width/2))
                centery = int(top+(height/2)) 
                # 取得したQRコードの範囲を描画
                # print(symbol.polygon)
                # NumPyのarray形式にする
                pts = np.array( symbol.polygon ) 
                cv2.polylines(img_qr, [pts], True, (0,255,0), thickness=3) 

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_qr,str_data,(centerx,centery), font, 2,(255,0,255),2,cv2.LINE_AA)
                
            plt.figure(figsize=(8,8))
            plt.imshow(img_qr)
            # plt.show()
            plt.pause(posetime) 

            if image_save == True:
            # 結果を保存
                cv2.imwrite('qr_image.png', img_qr)
                
    @staticmethod
    def find_rect(image_file,posetime=2,image_save=False):
        """
        輪郭を抽出する関数
        
        Note:
            図形認識(あまりよくない)
            OpenCV + Pythonで特定の図形を検出する１(図形の領域を矩形で取得) 
            https://symfoware.blog.fc2.com/blog-entry-2163.html
            OpenCV - findContours で輪郭抽出する方法    
            https://www.pynote.info/entry/opencv-findcontours
        """
        
        
        # ファイルを読み込み
        src = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        # 画像の大きさ取得
        height, width, channels = src.shape
        image_size = height * width
        # グレースケール化
        img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # しきい値指定によるフィルタリング
        retval, dst = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV )
        # 白黒の反転
    #     dst = cv2.bitwise_not(dst)
    #     # 再度フィルタリング
    #     retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     輪郭を抽出
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # この時点での状態をデバッグ出力
        dst = cv2.imread(image_file, cv2.IMREAD_COLOR)
        dst = cv2.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv2.LINE_AA)
        # cv2.imwrite('debug_1.png', dst)
        dst = cv2.imread(image_file, cv2.IMREAD_COLOR)
        for i, contour in enumerate(contours):
            # 小さな領域の場合は間引く
            area = cv2.contourArea(contour)
    #         if area < 500:
    #             continue
    #         # 画像全体を占める領域は除外する
    #         if image_size * 0.99 < area:
    #             continue
            
            # 外接矩形を取得
            x,y,w,h = cv2.boundingRect(contour)
            dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
        

        plt.figure(figsize=(8,8))
        plt.imshow(dst)
                # plt.show()
        plt.pause(posetime) 

        if image_save == True:
            # 結果を保存
            cv2.imwrite('result.png', dst)
                    
    
                
if __name__ == '__main__':
    image_file = './data'
    QrBase.qr_reader(image_file, posetime=2, image_save=False)
    