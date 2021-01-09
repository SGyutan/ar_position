#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real Time AR Marker Recognition

"""
import datetime
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2          

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


def image_transform(img, corners, ids, size=None, AR_ids_list=[], position_num=4, 
                    info=False, image_save=False, ar_cut_position='edge'):
    """Function to cut out an image from 4 markers

    Args:
        img (ndarray): image
        corners (ndarray): from return aruco.detectMarkers(image, dictionary)
        ids (ndarray): from return aruco.detectMarkers(image, dictionary)
        size (hight, width): example (150,150)  Defaults to None.
        AR_ids_list (list): 4 AR marker id example [0,1,2,3] . Defaults to [].
        position_num (int): indicator AR marker id. Defaults to 4.
        info (bool, optional): print out the results. Defaults to False.
        image_save (bool, optional): save image. Defaults to False.
        ar_cut_position (str): 'center' or 'edge. Defaults to 'edge'.

    Returns:
        image: ndarray or NoneType
        indicator position [x,y] :list or NoneType
        
    """

    if size== None:
        size = (150,150)
    
    if AR_ids_list == []:
            # LT-> RT -> RB -> LB
        AR_ids_list = [0,1,2,3]
    else:
        AR_ids_list = AR_ids_list
    
    AR_ids_list.append(position_num)
    
    # ids-> ndarray
    # ids and corners are sorted from low ids number.
    # change ndarry to list
    ids_list = ids.ravel().tolist()

    if len(ids) >= 4 and all(map(ids_list.__contains__, (AR_ids_list[:-1]))):
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
        it_AR_ids_list = AR_ids_list
        
        if len(ids) == 4:
            it_AR_ids_list = AR_ids_list[:-1]
        
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
        img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
        
        if info:
            comment = 'Find:{}'.format(ids_list)

        if image_save:
            out_file_name = 'trans_{}_{}'.format(current_datetime_str(),'trans.png')
            cv2.imwrite(str(out_file_name), img_trans) 
            
        if position_num in ids_list:
            
            # m[4]は(1x2)の行列、計算のために(1x3)の行列を作成->(m[4][0],m[4][1],1)
            indicator_cp = trans_mat.dot(np.array([m[4][0], m[4][1], 1]))
            # To recognize other corners
            corner01 = trans_mat.dot(np.array([m[0][0], m[0][1], 1]))
            corner02 = trans_mat.dot(np.array([m[1][0], m[1][1], 1]))
            corner03 = trans_mat.dot(np.array([m[2][0], m[2][1], 1]))
            corner04 = trans_mat.dot(np.array([m[3][0], m[3][1], 1]))

            # 3列目の値で割ると変換した値になる
            cal_indicatorx = indicator_cp[0]/indicator_cp[2]
            cal_indicatory = indicator_cp[1]/indicator_cp[2]
            cal_indicator = [cal_indicatorx,cal_indicatory]

            corner_list = [corner01, corner02, corner03, corner04, indicator_cp]
            
            if info:
                print('Position before calibrated x:{:.2f}, y:{:.2f}'.format(indicator_cp[0],indicator_cp[1]))
                print('Position x:{:.2f}, y:{:.2f}'.format(cal_indicatorx, cal_indicatory))
                print('corner codinate:[x,y,z] x=0 or height, y=0 or width, z=1 -> ideal=1')
                for ii in corner_list:
                    print('corner codinate: {}'.format(ii))
                    

            m_position = f'{int(cal_indicatorx)}, {int(cal_indicatory)}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_indicator = img_trans.copy() 
            
            # TODO：ホルダーの向きによって測定位置を示すマーカーの位置座標を変更する。
            cv2.line(img_indicator, (int(cal_indicatorx), height), (int(cal_indicatorx), 0), (0, 0, 0), thickness=2, lineType=cv2.LINE_4)
            cv2.putText(img_indicator,m_position,(20,20), font, 0.5,(0,255,0),1,cv2.LINE_AA)
            # cv2.putText(img_indicator,m_position,(int(cal_indicatorx),30), font, 0.5,(0,255,0),1,cv2.LINE_AA)
            
            if image_save:
                # write indicater into image
                out_file_name = 'Ind_{}_{}'.format(current_datetime_str(),'ind.png')
                cv2.imwrite(str(out_file_name),img_indicator) 
                
            return img_indicator, cal_indicator
        
        return img_trans, None
    
    else:
        if info:
            print('Recognized Markers:{}'.format(ids_list))
        return None, None


def live_find(camera_num=0):
    """Find AR marker

    Args:
        camera_num (int): camera number. Defaults to 0.
    """

    print('Quit -> Press "q"')
    
    aruco = cv2.aruco
    # AR marker(4x4 dot，ids until 50)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)   

    time.sleep(0.5)     
    
    cap = cv2.VideoCapture(camera_num)
    
    while True:
        
        ret,frame = cap.read()
        time.sleep(0.2)
        
        if ret == True:
            
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)      

            # detect AR marker and draw AR region and id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(image, dictionary) 
            aruco.drawDetectedMarkers(image, corners, ids, (0,255,0)) 
            
            cv2.imshow('Find AR', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def live_find_save(camera_num=0, time_out=10):
    """Find AR marker until finding over 4 markers or timeout

    Args:
        camera_num (int): camera number. Defaults to 0.
        time_out (int): time out time [s]. Defaults to 10.
    """

    print('Quit -> Press "q"')
    
    aruco = cv2.aruco
    # AR marker(4x4 dot，ids until 50)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)   

    current_time = time.time() 
    
    time.sleep(0.5)   
    
    cap = cv2.VideoCapture(camera_num)
    
    while True:
        
        ret,frame = cap.read()
        time.sleep(0.2)
        
        if ret == True:

            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

            corners, ids, rejectedImgPoints = aruco.detectMarkers(image, dictionary) 
            overlay_image = image.copy()
            aruco.drawDetectedMarkers(overlay_image, corners, ids, (0,255,0)) 
            
            cv2.imshow('Ids', overlay_image) 
            
            if ids is not None:
                ids_list = ids.ravel().tolist()
            
                if len(ids) >= 4:
                    # file save
                    print('Find AR markers', ids_list)
                    output_path = Path("./data")
                    output_path.mkdir(exist_ok=True)

                    out_file_name = 'AR_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                    out_path_file = output_path/out_file_name
                    cv2.imwrite(str(out_path_file), image) 
                    
                    out_file_name2 = 'ARO_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                    out_path_file2 = output_path/out_file_name2
                    cv2.imwrite(str(out_path_file2), overlay_image) 
                    time.sleep(0.5)
                    break
                
                else:
                    pass
                
            elasp_time = time.time()- current_time
            
            if elasp_time >= time_out :
                break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def live_find_transform(camera_num=0, time_out= 20, save_image=False):
    """Find AR marker until finding over 4 markers or timeout
        and estimate indicator position and transform the image. 

    Args:
        camera_num (int, optional): camera number. Defaults to 0.
        time_out (int, optional): time out time [s] if time_out=0, no time out. Defaults to 20.
        save_image (bool, optional): save image if find 5 marker. Defaults to False.
    """

    print('Quit -> press "q"')
    
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)   


    time.sleep(0.5) 
    current_time = time.time() 
    
    cap = cv2.VideoCapture(camera_num)
    
    while True:
        
        ret,frame = cap.read()
        time.sleep(0.2)
        
        if ret == True:

            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            corners, ids, rejectedImgPoints = aruco.detectMarkers(image, dictionary) 
            overlay_image = image.copy()
            aruco.drawDetectedMarkers(overlay_image, corners, ids, (0,255,0)) 
            
            cv2.imshow('AR marked', overlay_image)    
            # Display multiple windows at the same time. 
            # In that case, change the window name specified by the first argument to another name.
            
            if ids is not  None:
                ids_list = ids.ravel().tolist()
            
                if len(ids) >= 4:
                    img_trans,indicator = image_transform(img=image, corners=corners, ids=ids, 
                                            size=(150,150), AR_ids_list=[34,37,10,17], position_num=4, 
                                            info=False, image_save=False, ar_cut_position='edge')

                    if img_trans is not None:
                        # print(img_trans.shape[0],img_trans.shape[1])
                        # resized_img = cv2.resize(img,(width/2, height/2))
                        # cv2.namedWindow('Trans', cv2.WINDOW_NORMAL)
                        cv2.imshow('Trans', img_trans) 
                        print('position:',indicator)
                        
                        if len(ids) == 5 and save_image ==True:
                            # print('position:',indicator)
                            output_path = Path("./data")
                            output_path.mkdir(exist_ok=True)

                            out_file_name = 'AR_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                            out_path_file = output_path/out_file_name
                            cv2.imwrite(str(out_path_file), image) 
                            
                            out_file_name2 = 'ARO_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                            out_path_file2 = output_path/out_file_name2
                            cv2.imwrite(str(out_path_file2), img_trans) 
                            break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                elasp_time = time.time()- current_time
            
                if elasp_time >= time_out :
                    if time_out == 0:
                        pass
                    else:
                        break
                
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":  
    # live_find()    
    # live_find_save()
    live_find_transform(camera_num=0, time_out= 0, save_image=True)

