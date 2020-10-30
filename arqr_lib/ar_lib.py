"""
AR marker用のコード

20200501作成

ref:
[Python+OpenCVで、カメラ画像から机上の物体位置（実座標系）を計測してみる]
(https://qiita.com/code0327/items/c6e468da7007734c897f)
[ARマーカー認識プログラム]
(https://qiita.com/hsgucci/items/37becbb8bfe04330ce14)
[Python, OpenCVで幾何変換（アフィン変換・射影変換など）]
(https://note.nkmk.me/python-opencv-warp-affine-perspective/)

https://watlab-blog.com/2019/06/01/projection-transform/

"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ArBase():
    """
    AR codeの関数をまとめたクラス。Staticmethodのみ


    """

    @staticmethod
    def image_show(image_file, posetime=2, image_save=False):
        """
        Imgeを表示する関数

        example:
        image_file = 'target.png'
        posetime=2　表示する時間
        image_save=False　Saveするかしないか　Saveの時はTrue
        saveファイル名は、'result.png'　で固定されている

        """
        src = cv2.imread(image_file, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

        plt.figure(figsize=(8,8))
        plt.imshow(img_gray) 
        # plt.show()
        #指定した時間表示した後、自動で閉じて次のプログラムを実行
        plt.pause(posetime) 

        if image_save == True:
            # 結果を保存
            cv2.imwrite('result.png', img_gray)


    @staticmethod
    def qr_reader(image_file, posetime=2, image_save=False):
        """
        Opencvを使ったQRコードの認識
        QRコードが画面に一つの時のみ有効、それ以外はエラーになる。

                image_file = './photos/test003.jpg'
        # image_file = 'photos/QR_id1.png'

        """
        # print(cv2.__version__)
        img = cv2.imread(image_file)
        # plt.imshow(img)
        # plt.pause(1) 
        # print(type(img))

        qr = cv2.QRCodeDetector()
        # print(qr.detectAndDecode(img))
        data, points, straight_qrcode = qr.detectAndDecode(img)
        if data:
            print(f'decoded data: {data}')
            for i in range(4):
                cv2.line(img, tuple(points[i][0]), tuple(points[(i + 1) % len(points)][0]), (0, 0, 255), 4)
            plt.title(f'decoded data: {data}')
            plt.figure(figsize=(8,8))
            plt.imshow(img)
            # plt.show()
            plt.pause(2) 

            if image_save == True:
                # 結果を保存
                cv2.imwrite('opencv_qr.png', img)

        else:
            print('No recoganzize')

        print('Data:', data)
        print('Cordinates:',points)
        print(f'QR code version: {((straight_qrcode.shape[0] - 21) / 4) + 1}')


    @staticmethod
    def ar_marker_make(variety_num=4,output_dir='./marker'):
        """
        markar画像を作成する関数

        param:
        variety_num：markerの作成種類（個数）　defult = 4
        output_dir: outputdir defult ='./marker'
        
        Note:コード作成メモ
        aruco.getPredefinedDictionary(...) :
        事前定義されたマーカーが格納されている辞書を取得
        引数 aruco.DICT_4X4_50 :
        正方形の内部に 4×44×4 の塗りつぶしパターンを持った
        マーカーが最大 5050 個まで利用可能な辞書を選択
        
        """
        if output_dir == None:
            output_dir='./marker'

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        aruco = cv2.aruco
        p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        marker =  [0] * variety_num # 初期化
        for i in range(len(marker)):
            marker[i] = aruco.drawMarker(p_dict, i, 75) # 75x75 px
            output_file_name = output_path.joinpath('marker{}.png'.format(i))
            cv2.imwrite(str(output_file_name), marker[i])

    @staticmethod
    def ar_find_overay(image_file, posetime=2, image_save=False):
        """
        ARコードを認識して、表示する関数。
        find AR and overay the results

        params:
        image_file:
        posetime 表示する時間: int  (ex:posetime=2)　表示する時間
        image_saveSaveするかしないか: False　or True
        saveファイル名は、'AR_image.png'　で固定されている

        return:
            overrayed image
        """
        aruco = cv2.aruco
        p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        image_path = Path(image_file)
        image_basename = image_path.name
        
        imgAR = cv2.imread(str(image_path))
        try:
            corners, ids, rejectedImgPoints = aruco.detectMarkers(imgAR, p_dict) # 検出
            img_marked = aruco.drawDetectedMarkers(imgAR.copy(), corners, ids)   # 検出結果をオーバーレイ
            # print(corners)
            # print(ids)
            print('number of AR finds: ', len(ids))


            plt.figure(figsize=(6,6))
            plt.title(image_basename)
            plt.imshow(img_marked) 
            # plt.show()
            #指定した時間表示した後、自動で閉じて次のプログラムを実行
            plt.pause(posetime) 

            if image_save == True:
                # 結果を保存
                cv2.imwrite('ARimage.png', img_marked)
            
            return img_marked
        
        except:
            print('No detect')
            return None

    @staticmethod
    def real_size_ar(image_file,output_path='./data',size=None,ar_cut_position='edge'):
        """
        4つのマーカーから画像を切り出す関数
        （1）マーカー座標を取得
        （2）射影変換にて真上から見た画像に変換、切り出す。

        param:
        image_file：str or pathlib object
        output_path: None => defult './data'
        size: 変形後画像サイズ tuple (width, height) 
        ar_cut_position:'edge' or 'center'
        
        example:
        image_file='WIN_20200403_15_00_41_Pro.jpg'
        size = (500,500)
        ar_cut_position='edge' or 'center'

        """
        if output_path== None:
            output_path='./data'
            
        out_image_path = Path(output_path)
        out_image_path.mkdir(exist_ok=True)
            
        if size== None:
            size = (500,500)

        aruco = cv2.aruco
        p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        
        image_path = Path(image_file)
        image_basename = image_path.name
        
        # image_file='WIN_20200403_15_00_41_Pro.jpg' 
        img = cv2.imread(str(image_path))
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

            width, height = size[0],size[1] # 変形後画像サイズ

            marker_coordinates = np.float32(m)
            true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
            #射影変換
            trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
            img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
            

            out_file_name = 'marker_{}'.format(image_basename)
            out = out_image_path/out_file_name
            
            cv2.imwrite(str(out), img_trans)
            
            plt.figure(figsize=(6,6))
            plt.title(image_basename)
            plt.imshow(img_trans)
            # plt.show()
            plt.pause(1) #指定した時間表示した後、自動で閉じて次のプログラムを実行
            return img_trans
        
        except:
            print('No recoganzize')
            return None
    
    @staticmethod
    def real_size_ar_with_indicator(image_file,output_path='./data',size= (150,150),ar_cut_position='edge'):
        """
        4つのマーカーを使って、試料の場所を推定する。
        5つ目のマーカー座標を推定する。
            (1)マーカー座標を取得
            (2)射影変換にて真上から見た画像に変換、切り出す。
        
        param:
        image_file：str or pathlib object
        output_path: None => defult './data'
        size: 変形後画像サイズ tuple (width, height) 
        ar_cut_position:'edge' or 'center'
        
        Note:
        うまくコーナーが見つけられないとpass
        
        example:
        image_file='WIN_20200403_15_00_41_Pro.jpg'
        size=(500,500)
        _ = real_size_ar_with_indicator(image_file=img,output_path=None,size=(200,200),ar_cut_position='edge')

        """
        if output_path== None:
            output_path='./data'
            
        out_image_path = Path(output_path)
        out_image_path.mkdir(exist_ok=True)    
        
        if size== None:
            size = (150,150)

        aruco = cv2.aruco
        p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        
        image_path = Path(image_file)
        image_basename = image_path.name
        
        
        img = cv2.imread(str(image_path))
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) # 検出
    #     print(corners)
        try:
            # 時計回りで左上から順にマーカーの「中心座標」を m に格納
            #ravel関数は、多次元のリストを1次元のリストにする
            m = np.empty((5,2))
            if ar_cut_position == 'center':
                for i,c in zip(ids.ravel(), corners):
                    print('i',i)
                    print('c',c)
                    m[i] = c[0].mean(axis=0)
                    # print('id: {}, mi: {}'.format(i,m[i]))
            
            elif ar_cut_position == 'edge':
                # print('edge')
                corners2 = [np.empty((1,4,2))]*5
                # corners　->（1,4,2）　c[0]に4角の座標が入っている。
                for i,c in zip(ids.ravel(), corners):
                    corners2[i] = c.copy()
                    m[0] = corners2[0][0][2]
                    m[1] = corners2[1][0][3]
                    m[2] = corners2[2][0][0]
                    m[3] = corners2[3][0][1]
                    m[4] = corners2[4][0].mean(axis=0)
                    # print('id: {}, mi: {}'.format(i,m[i]))

            else:
                pass
            
    #         print(m[4])
            width, height = size[0],size[1] # 変形後画像サイズ

            marker_coordinates = np.float32(m[0:4])
            true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
            #射影変換
            # (3x3)の変換行列
            trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
            img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
            
    #         print(trans_mat,type(trans_mat))
            #m[4]は（1ｘ2）の行列　（1ｘ3）の行列を作成　（m[4][0],m[4][1],1)
            # cv2.imwrite('marker2_{}'.format(image_file),img_trans)

            indicater_cp=trans_mat.dot(np.array([m[4][0],m[4][1],1]))
            print('Indicater AR Current Position : ',indicater_cp)
            # 他の座標も確認のために
            corner01=trans_mat.dot(np.array([m[0][0],m[0][1],1]))
            corner02=trans_mat.dot(np.array([m[1][0],m[1][1],1]))
            corner03=trans_mat.dot(np.array([m[2][0],m[2][1],1]))
            corner04=trans_mat.dot(np.array([m[3][0],m[3][1],1]))
            corner_list= [corner01,corner02,corner03,corner04]
            print('corner codinate:[x,y,1] x=0 or height, y=0 or width')
            for ii in corner_list:
                print('corner codinate: {}'.format(ii))

            #画像に位置を書き込む場合コメントアウトを外す
    #         indicater_codi = (int(indicater_cp[0]),int(indicater_cp[1]+50))
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(img_trans,m_position,indicater_codi, font, 1,(0,100,0),2,cv2.LINE_AA)

            
            out_file_name = 'marker_{}'.format(image_basename)
            out = out_image_path/out_file_name
            
            cv2.imwrite(str(out), img_trans)
            
            plt.figure(figsize=(6,6))
            plt.title(image_basename)
            plt.imshow(img_trans)
            # plt.show()
            #指定した時間表示した後、自動で閉じて次のプログラムを実行
            plt.pause(2) 
            return img_trans
        
        except:
            print('No recoganzize')
            return None

if __name__ == "__main__":
    image_file='./photos/wb1.JPG'
    outpath='./cutout'
    
    ArBase.image_show(image_file, posetime=2, image_save=False)
    # ArBase.qr_reader(image_file, posetime=2, image_save=False)
    # ArBase.ar_marker_make(variety_num=4,output_dir='./marker')
    ArBase.ar_find_overay(image_file, posetime=2, image_save=False)
    # ArBase.real_size_ar(image_file,output_path='./data',size=None,ar_cut_position='edge')
    ArBase.real_size_ar_with_indicator(image_file,output_path='./data',size= (150,150),ar_cut_position='edge')
    
    
