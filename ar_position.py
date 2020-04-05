"""
sample holderにARコードを貼り付けて、ARコードが写っている画像から、
実長さに変換して切り出す

"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def real_size_ar(image_file,output_path=None,size=None,ar_cut_position='edge'):
    """
    #真上から見た画像に変換
    #うまくコーナーが見つけられないとpass
    
    image_file: image file path
    ex) image_file='WIN_20200403_15_00_41_Pro.jpg'
    size: 変形後画像サイズ 
    ex) size=500
    
    ar_cut_position='edge' or 'center'
    """
    if output_path== None:
        output_path='./data'
        
    if size== None:
        size = 500

    aruco = cv2.aruco
    p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    image_path = Path(image_file)
    image_basename = image_path.name
    
    # image_file='WIN_20200403_15_00_41_Pro.jpg' 
    img = cv2.imread(image_file)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) # 検出
    # print(corners)
    try:
        # 時計回りで左上から順にマーカーの「中心座標」を m に格納
        m = np.empty((4,2))
        if ar_cut_position == 'center':
            for i,c in zip(ids.ravel(), corners):
                m[i] = c[0].mean(axis=0)
        
        elif ar_cut_position == 'edge':
            corners2 = [np.empty((1,4,2))]*4
            for i,c in zip(ids.ravel(), corners):
                corners2[i] = c.copy()
                m[0] = corners2[0][0][2]
                m[1] = corners2[1][0][3]
                m[2] = corners2[2][0][0]
                m[3] = corners2[3][0][1]
        
        else:
            pass

        width, height = (size,size) # 変形後画像サイズ

        marker_coordinates = np.float32(m)
        true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
        #射影変換
        trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
        img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
        
        out_image_path = Path(output_path)
        out_image_path.mkdir(exist_ok=True)
        out_file_name = 'marker_{}'.format(image_basename)
        out = out_image_path/out_file_name
        
        cv2.imwrite(str(out), img_trans)
        
        plt.imshow(img_trans)
        # plt.show()
        plt.pause(1) #指定した時間表示した後、自動で閉じて次のプログラムを実行
        return img_trans
    
    except:
        print('No recoganzize')
        return None

    
def ar_maker_maker():
    """
    aruco.getPredefinedDictionary(...) :
    事前定義されたマーカーが格納されている辞書を取得
    
    引数 aruco.DICT_4X4_50 :
    正方形の内部に 4×44×4 の塗りつぶしパターンを持った
    マーカーが最大 5050 個まで利用可能な辞書を選択
    
    """
    
    aruco = cv2.aruco
    p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker =  [0] * 4 # 初期化
    for i in range(len(marker)):
        marker[i] = aruco.drawMarker(p_dict, i, 75) # 75x75 px
        cv2.imwrite(f'marker{i}.png', marker[i])
        
if __name__ == "__main__":
    img='./photos/WIN_20200318_17_03_04_Pro.jpg'
    outpath='./cutout'
    real_size_ar(image_file=img,output_path=outpath,size=None,ar_cut_position='edge')