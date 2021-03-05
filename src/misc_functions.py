        
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):
        print('IN..')
        exit()
        
                    resize_imface = resize_im(image_face,160)
            print(f'[INFO] image shape - {image_face.shape}')
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),10)
        for key,points in landmarks.items():
            cv2.circle(image,points,3,(0,0,255),30)
            save_as = str(i)+'.jpg'
            save_in = os.path.join("recognised_faces",save_as) 
        print(f'[INFO] Saving image in {save_in}..')
        cv2.imshow('test',resize_imface)
        cv2.imwrite(save_in,resize_imface)
        
        
        def make_weights(images,nclasses):
    count = [0]*nclasses
    for item in images:
        count[item[1]] +=1 
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
        
    weight = [0]* len(images)
    for idx,val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Class Activation Maps
# b = weights[pred_labs[0]]
# a = blob.squeeze().reshape(1792,9)
# print(f'[INFO] weights shape - {b.shape}')
# print(f'[INFO] reshaping features - {a.shape}')

# cam = np.matmul(b,a)
# print(f'[INFO] CAM features - {cam.shape}')

# cam.reshape(3,3)

# cam = cam-np.min(cam)
# cam_img = cam/np.max(cam)
# cam_img = np.uint8(255*cam_img)
# op_cam = cv2.resize(cam_img,(256,256))

# heatmap = cv2.applyColorMap(cv2.resize(op_cam,(160,160)),cv2.COLORMAP_JET)
# print(f'[INFO] heatmap shape - {heatmap.shape}')
# plt.imshow(heatmap.astype('uint8'));