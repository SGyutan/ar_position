"""
20210107に修正
・4角に配置するARマーカーのIDを指定できるようにする。
これまで、ARのIDは左上0番、右上1番、右下2番、左下3番、測定位置4番
LT-> RT -> RB -> LB [0,1,2,3]
ホルダーへの加工を考えよりシンプルなマーカーを選ぶ。
combi sample holder [34,37,10,17]


20200710修正
・class に変更

200807作成
・AR marker用のコード
・自動認識
・ARマーカーの認識　ホルダー位置のみ、ホルダー位置+インジケータ
・認識がうまく行かない時の対応
・アファイン変換
・測定位置の推定

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

import datetime

def current_datetime_str(type=3):
    """
    Returns the current date and time as a string
    type1: "%Y-%m-%d %H:%M:%S"
    type2: "%Y%m%d%H%M%S"
    type3: "%Y%m%d_%H%M%S"
    type4: "%Y%m%d%H%M"

    elae: "%Y%m%d"

    :return:
    """
    now = datetime.datetime.now()
    if type == 1:
        now_string = now.strftime("%Y-%m-%d %H:%M:%S")
    elif type == 2:
        now_string = now.strftime("%Y%m%d%H%M%S")
    elif type == 3:
        now_string = now.strftime("%Y%m%d_%H%M%S")
    elif type == 4:
        now_string = now.strftime("%Y%m%d%H%M")
    elif type == 5:
        now_string = now.strftime("%m%d_%H:%M:%S")
    elif type == 6:
        now_string = now.strftime("%Y%m%d")    
    else:
        now_string = now

    return  now_string

class ArFind():
    """
    AR find and then transform the image

    """

    def __init__(self, image_file, np_image=None,size=(150, 150), output_path='./data', 
                 holder_AR_ids_list=[],position_num=4):
        """
        image_file: input file name
        size: after size,  tuple (width, height) 
        marker_num_list: AR marker ID list. 
            exampel  # LT-> RT -> RB -> LB [0,1,2,3]
            combi sample holder [34,37,10,17]
        position_num: measurment position AR ID   
        """
        self.image_file = Path(image_file)
        self.image_basename = self.image_file.name
        self.image_basestem = self.image_file.stem
        self.imag_parent = self.image_file.parent  
        
        self.size = size
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        if holder_AR_ids_list == []:
            # LT-> RT -> RB -> LB
            self.holder_AR_ids_list = [0,1,2,3]
        else:
            self.holder_AR_ids_list = holder_AR_ids_list
        self.position_num = position_num
        self.holder_AR_ids_list.append(position_num)
        
        if np_image == None:
            self.imgAR = cv2.imread(str(self.image_file), cv2.IMREAD_COLOR)
            
        else:
            self.imgAR = np_image
        
        # Change Gray scale
        self.imgAR_gray = cv2.cvtColor(self.imgAR, cv2.COLOR_RGB2GRAY)
            
        # aruco instance
        self.aruco = cv2.aruco
        self.p_dict = self.aruco.getPredefinedDictionary(self.aruco.DICT_4X4_50)

    def fig_plot(self, np_image, comment='', figsize=(8, 8), pause=2):
        """
        Args:
            np_image (numpy array): image file to ndarray
            comment (str, optional): [description]. Defaults to ''.
            figsize (tuple, optional): [description]. Defaults to (8, 8).
            pause (int, optional): [description]. Defaults to 2.
        """
        print(comment)
        plt.figure(figsize=figsize)
        title_input = '{} {}'.format(self.image_basename, comment)
        plt.title(title_input)
        plt.imshow(np_image)
        # plt.show()
        # 指定した時間表示した後、自動で閉じて次のプログラムを実行
        plt.pause(pause)

        
    def save_image(self, nd_image, out_file_name):
        out_path_file = self.output_path/out_file_name
        cv2.imwrite(str(out_path_file), nd_image) 

    def find_ids(self):
        """
        Recognize the AR code and return the number of recognized ARs

        return:
            n_find_labels
        """

        try:
            corners, ids, rejectedImgPoints = self.aruco.detectMarkers(
                self.imgAR_gray, self.p_dict) 

            n_find_labels = len(ids)
            print(self.image_basename)
            print('number of AR finds: ', n_find_labels)
            print('AR IDs: ',ids.ravel().tolist())

            return n_find_labels

        except:
            print('No detect')
            return 0

    def find_ids_overlay(self,info=True,img_save=False):
        """
        find AR and overlay the results

        return:
                img_marked, corners, ids
        """

        try:
            corners, ids, rejectedImgPoints = self.aruco.detectMarkers(
                self.imgAR_gray, self.p_dict) 
            
            # overlay the results of ids and rectangular
            img_marked = self.aruco.drawDetectedMarkers(
                self.imgAR.copy(), corners, ids)   
            # print(corners)
            # print(ids)
            n_find_labels = len(ids)
            
            if info:
                print('number of AR finds: ', n_find_labels)
                print('AR IDs: ',ids.ravel().tolist())
                comment = 'Find:{}'.format(n_find_labels)
                self.fig_plot(img_marked, comment, figsize=(8, 8))
                
            if img_save:
                out_file_name = 'AR_{}_{}'.format(current_datetime_str(),self.image_basename)
                self.save_image(img_marked, out_file_name)

            return img_marked, corners, ids

        except:
            print('No detect')
            return None, None, None

    def qr_reader(self):
        """
        Opencvを使ったQRコードの認識
        QRコードが画面に一つの時のみ有効、それ以外はエラーになる。

        """
        # print(cv2.__version__)

        qr = cv2.QRCodeDetector()
        # print(qr.detectAndDecode(img))
        data, points, straight_qrcode = qr.detectAndDecode(self.imgAR_gray)
        img_qr = self.imgAR_gray.copy()
        if data:
            print(f'decoded data: {data}')
            for i in range(4):
                cv2.line(img_qr, tuple(points[i][0]), tuple(
                    points[(i + 1) % len(points)][0]), (0, 0, 255), 4)

            if self.plot_flag == True:
                self._plot(img_qr, f'decoded data: {data}', figsize=(8, 8))

            out_file_name = 'qr_{}'.format(self.image_basename)
            out = self.output_path/out_file_name
            cv2.imwrite(str(out), img_qr)

            # print('Data:', data)
            # print('Cordinates:',points)
            # print(f'QR code version: {((straight_qrcode.shape[0] - 21) / 4) + 1}')

            return out_file_name, img_qr

        else:
            print('No Detection')
            return None, None

    def image_conversion(self,size=None, info=True, image_save=True, ar_cut_position='edge'):
        """ Function to cut out an image from 4 markers
        Args:
            size ([type], optional): [description]. Defaults to None.
            info (bool, optional): [description]. Defaults to True.
            image_save (bool, optional): [description]. Defaults to True.
            ar_cut_position (str, optional): 'edge' or 'center'. Defaults to 'edge'.

        Returns:
            [type]: [description]

        """

        if size== None:
            size = (150,150)
            
        img_marked, corners, ids = self.find_ids_overlay(info=False,img_save=False)
        
        if ids is not None:
            ids_list = ids.ravel().tolist()
            # ids-> ndarray
            # ids and corners are sorted from low ids number.
            # change ndarry to list

            if len(ids) >= 4 and all(map(ids_list.__contains__, (self.holder_AR_ids_list[:-1]))):
                # all... means -> AR_ids_list[0] in AR_ids_list and AR_ids_list[1] inAR_ids_list 
                # Line up clockwise
                # LT-> RT -> RB -> LB
                # + 0,1 +
                # + 3,2 + -> [0,1,2,3]
                # New version
                # + 34,37 +
                # + 17,10 + -> [34,37,10,17]

                # store center position of each clockwise from left-top into m array
                # array dimension -> 5raw (ids) (x,y) => (5,2)
                
                it_AR_ids_list = self.holder_AR_ids_list
                if len(ids) == 4:
                    it_AR_ids_list = self.holder_AR_ids_list[:-1]
                    
                m = np.empty((5,2))
                
                
                if ar_cut_position == 'center':
                    for i, id_num in enumerate(it_AR_ids_list):
                        m[i] = corners[ids_list.index(id_num)][0].mean(axis=0)
                
                elif ar_cut_position == 'edge':
                    # 5raw(ids), 4corners, (x,y) -> (5,4,2)
                    corners2 = [np.empty((1,4,2))]*5
                    for i, id_num in enumerate(it_AR_ids_list):
                        corners2[i] = corners[ids_list.index(id_num)].copy()
                        m[0] = corners2[0][0][2]
                        m[1] = corners2[1][0][3]
                        m[2] = corners2[2][0][0]
                        m[3] = corners2[3][0][1]
                        m[4] = corners2[4][0].mean(axis=0)
                
                else:
                    pass
                # Image size after transformation
                width, height = size[0],size[1] 

                marker_coordinates = np.float32(m[0:4])
                # LT(0,0)-> RT -> RB(width,height) -> LB
                true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
                
                # Homography
                trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
                img_trans = cv2.warpPerspective(self.imgAR,trans_mat,(width, height))
                
                if info:
                    comment = 'Find:{}'.format(ids_list)
                    self.fig_plot(img_trans, comment, figsize=(8, 8))

                if image_save:
                    out_file_name = 'Mark_{}_{}'.format(current_datetime_str(),self.image_basename)
                    self.save_image(img_trans, out_file_name)
                    
                if self.position_num in ids_list:
                    
                    # m[4]は(1x2)の行列、計算のために(1x3)の行列を作成->(m[4][0],m[4][1],1)
                    indicator_cp = trans_mat.dot(np.array([m[4][0], m[4][1], 1]))
                    # 逆行列
                    # https://qiita.com/naosk8/items/cde89dd93044e0abb054
                    # inv_ind = np.dot(np.linalg.inv(trans_mat),
                    #                  np.array([m[4][0], m[4][1], 1]))
                    # inv_ind = np.dot(trans_mat, np.array([m[4][0], m[4][1], 1]))
                    # print(inv_ind)

                    # print('Indicator AR Current Position : ', indicator_cp)

                    # To recognize other corners
                    corner01 = trans_mat.dot(np.array([m[0][0], m[0][1], 1]))
                    corner02 = trans_mat.dot(np.array([m[1][0], m[1][1], 1]))
                    corner03 = trans_mat.dot(np.array([m[2][0], m[2][1], 1]))
                    corner04 = trans_mat.dot(np.array([m[3][0], m[3][1], 1]))

                    # ！！ 3列目の値で割ると変換した値になる（拡大縮小の効果も入れる必要がある。）
                    cal_indicatorx = indicator_cp[0]/indicator_cp[2]
                    cal_indicatory = indicator_cp[1]/indicator_cp[2]
                    cal_indicator = [cal_indicatorx,cal_indicatory]

                    # inv_cor01 = np.dot(np.linalg.inv(trans_mat),
                    #                    np.array([m[0][0], m[0][1], 1]))
                    # inv_cor02 = np.dot(np.linalg.inv(trans_mat),
                    #                    np.array([m[1][0], m[1][1], 1]))
                    # inv_cor03 = np.dot(np.linalg.inv(trans_mat),
                    #                    np.array([m[2][0], m[2][1], 1]))
                    # inv_cor04 = np.dot(np.linalg.inv(trans_mat),
                    #                    np.array([m[3][0], m[3][1], 1]))
                    # inv_corner_list = [inv_cor01, inv_cor02, inv_cor03, inv_cor04]
                    # for ii in inv_corner_list:
                    #     print('inv_corner codinate: {}'.format(ii))
                    corner_list = [corner01, corner02, corner03, corner04, indicator_cp]
                    
                    if info:
                        print('Position before calibrated x:{:.2f}, y:{:.2f}'.format(indicator_cp[0],indicator_cp[1]))
                        print('Position x:{:.2f}, y:{:.2f}'.format(cal_indicatorx, cal_indicatory))
                        print('corner codinate:[x,y,z] x=0 or height, y=0 or width, z=1 -> ideal=1')
                        for ii in corner_list:
                            print('corner codinate: {}'.format(ii))
                            
                    if image_save:
                        # write indicater into image
                        m_position = f'{int(cal_indicatorx)}, {int(cal_indicatory)}'
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        img_indicator = img_trans.copy() 
                        
                        # TODO：ホルダーの向きによって測定位置を示すマーカーの位置座標を変更する。
                        cv2.line(img_indicator, (int(cal_indicatorx), height), (int(cal_indicatorx), 0), (0, 0, 0), thickness=2, lineType=cv2.LINE_4)
                        cv2.putText(img_indicator,m_position,(int(cal_indicatorx),30), font, 0.5,(0,255,0),1,cv2.LINE_AA)
                        out_file_name = 'Ind_{}_{}'.format(current_datetime_str(),self.image_basename)
                        self.save_image(img_indicator, out_file_name)
                        
                    return img_trans, cal_indicator
            
            return img_trans, None
        
        else:
            # print('Recognized Markers:{}'.format(ids_list))
            return None, None


    @staticmethod
    def ar_marker_generator(variety_num=4, output_dir='./marker'):
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
            output_dir = './marker'

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        aruco = cv2.aruco
        p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        marker = [0] * variety_num  # initialize
        for i in range(len(marker)):
            marker[i] = aruco.drawMarker(p_dict, i, 75)  # 75x75 px
            output_file_name = output_path.joinpass('marker{}.png'.format(i))
            cv2.imwrite(output_file_name, marker[i])
            
            
if __name__ == "__main__":

    image_file = './photos/ar2.PNG'
    artest = ArFind(image_file, size=(150, 150), output_path='./data', 
                    holder_AR_ids_list=[34,37,10,17],position_num=4, plot_flag=True)
    artest.find_ids()
    artest.find_ids_overlay(info=True,img_save=True)
    a,b = artest.image_conversion(size=(150,150),info=True, image_save=True, ar_cut_position='edge')
    print(a)
    print(b)
    
