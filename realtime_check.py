#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real Time AR Marker Recognition

"""
import numpy as np
import matplotlib.pyplot as plt

import time         
import cv2          

# メイン関数
def main():
    # OpenCVが持つARマーカーライブラリ「aruco」を使う
    print('Quit -> press "q"')
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)   # ARマーカーは「4x4ドット，ID番号50まで」の辞書を使う

    current_time = time.time()  # 現在時刻の保存変数
    pre_time = current_time     # 5秒ごとの'command'送信のための時刻変数

    time.sleep(0.5)     # 通信が安定するまでちょっと待つ
    
    cap = cv2.VideoCapture(0)
    
    while True:
        
        ret,frame = cap.read()
        time.sleep(0.2)
        
        if ret == True:
            # (B)ここから画像処理
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)      # OpenCV用のカラー並びに変換する


            # ARマーカーの検出と，枠線の描画
            corners, ids, rejectedImgPoints = aruco.detectMarkers(image, dictionary) #マーカを検出
            aruco.drawDetectedMarkers(image, corners, ids, (0,255,0)) #検出したマーカ情報を元に，原画像に描画する
            
            cv2.imshow('OpenCV Window', image)    # ウィンドウに表示するイメージを変えれば色々表示できる
             
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":      # importされると"__main__"は入らないので，実行かimportかを判断できる．
    main()    # メイン関数を実行

