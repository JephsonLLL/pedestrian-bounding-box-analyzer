from re import I
import cv2
import os
import numpy as np
from itertools import cycle

class PedTrackWin:
   def __init__(self, window_title='Ped Track', data_image_path = './train_boxes_numbered/0002', box_path = 'box.txt'):
      self.window_title = window_title
      self.rect_p1 = None
      self.rect_p2 = None
      self.bounding_boxes = None
      self.frame_box = None
      self.ped_count = 0
      self.numof_ped_in_rect = 0
      self.img = None
      self.frame = 0
      self.video_len = 0
      self.total_peds = 0
      self.curr_ped_arr = None
      self.enter_map = {}
      self.leave_map = {}
      self.group_label = None
      self.group_member = None
      self.group_state = {}
      self.peds = []
      self.peds_set = set()
      self.form_and_dest = None
      
      # input image, use iterator
      self.filenames = os.listdir(data_image_path)
      self.img_iter = cycle([(cv2.imread(os.sep.join([data_image_path, x])), x) for x in self.filenames])

      # 2.1 count of all unique pedestrians
      with open(box_path, 'r') as f:
         self.bounding_boxes = f.read().splitlines()

      self.video_len = len(self.bounding_boxes)
      for row in self.bounding_boxes:
         ped_in_frame_arr = np.array(row.split(',')[1:]).reshape(-1,5)
         peds_in_frame = {int(row[0]) for row in ped_in_frame_arr}
         self.peds.append(peds_in_frame)
         for ped in peds_in_frame:
            self.peds_set.add(ped)

      self.total_peds = 400
      for i in range(1, self.total_peds+1):
         self.enter_map[i] = 0
         self.leave_map[i] = 0
         
      self.divide_group()
      # 3.1 how many pedestrians walk in groups and how many walk alone
      ped_in_group = set()
      for group in self.group_state:
         if len(self.group_state[group]) > 50:
            ped_in_group.update(group)

      # 3.2 Show occurrences of group formation and group destruction
      self.group_form_and_dest()
     
      key = 0
      is_pausing = False
      while key & 0xFF != 27:
         
         #cv2.namedWindow(self.window_title, cv2.WINDOW_FREERATIO )
         cv2.namedWindow(self.window_title)

         if not is_pausing:
            self.img, image_name = next(self.img_iter)
            self.frame  = int(image_name[0:-4])
            self.frame_box = self.bounding_boxes[self.frame-1].split(',')
            self.curr_ped_arr = np.array(self.frame_box[1:]).reshape(-1,5)

            # 2.2 count of pedestrians present in the current frame
            self.ped_count = int((len(self.frame_box) - 1)/5)

            self.group_visual()

            # 3.3 entering or leaving the scene
            self.ped_enter()
            self.ped_leave()

         if self.rect_p1 and self.rect_p2:
            # 2.4 count of pedestrians in rectangle
            self.count_rect_ped()
         else:
            self.numof_ped_in_rect = 0

         img_copy = self.img.copy()
         # 2.3 draw rectangle
         cv2.setMouseCallback(self.window_title, self.draw_rect)
         if self.rect_p1 and self.rect_p2:
            cv2.rectangle(img_copy, self.rect_p1, self.rect_p2, (0,0,255), 3)

         cv2.putText(img_copy, f'Total Peds: {len(self.peds_set)}', (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_8)
         cv2.putText(img_copy, f'Group Peds: {len(ped_in_group)}', (20,75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_8)
         cv2.putText(img_copy, f'Alone Peds: {len(self.peds_set)-len(ped_in_group)}', (20,110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_8)
         cv2.putText(img_copy, f'Peds in Frame: {self.ped_count}', (20,145), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_8)
         cv2.putText(img_copy, f'Peds in Rect: {str(self.numof_ped_in_rect)}', (20,180), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_8)
         cv2.putText(img_copy, str(self.frame), (1840,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_8)
         # click 'X'(may not work)
         #if cv2.getWindowProperty(self.window_title,1) == -1:
         #   break
         cv2.imshow(self.window_title, img_copy)
         key = cv2.waitKey(30)
         # click space to pause
         if key & 0xFF == 32:
            if is_pausing:
               is_pausing = False
            else:
               is_pausing = True
      cv2.destroyAllWindows()

   def draw_rect(self, event, x, y, flags, param):
      if event == cv2.EVENT_LBUTTONDOWN:
         self.rect_p2 = None
         self.rect_p1 = (x, y)
      elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
         self.rect_p2 = (x, y)
      elif event == cv2.EVENT_LBUTTONUP:
         self.rect_p2 = (x, y)
         if abs(self.rect_p1[0]-self.rect_p2[0])<5:
            self.rect_p1 = None
            self.rect_p2 = None

   def count_rect_ped(self):
      min_x = min(self.rect_p1[0], self.rect_p2[0])
      max_x = max(self.rect_p1[0], self.rect_p2[0])
      min_y = min(self.rect_p1[1], self.rect_p2[1])
      max_y = max(self.rect_p1[1], self.rect_p2[1])
      self.numof_ped_in_rect = 0

      for ped in self.curr_ped_arr:
         center_x = (int(ped[1]) + int(ped[3]))/2
         center_y = (int(ped[2]) + int(ped[4]))/2
         if center_x > min_x and center_x < max_x and center_y > min_y and center_y < max_y:
            self.numof_ped_in_rect = self.numof_ped_in_rect + 1

   def is_at_boundary(self, ped, frame_num):
      t = 40
      for row in np.array(self.bounding_boxes[frame_num-1].split(',')[1:]).reshape(-1,5):
         if int(row[0]) == ped:
            left = int(row[1])
            top = int(row[2])
            right = int(row[3])
            bottom = int(row[4])
            if left < t or top < t or right > 1920 - t or bottom > 1080 - t:
               return True
      return False

   def ped_leave(self):
      k = 10
      if self.frame < self.video_len-16-k:
         futu_peds = self.peds[self.frame+16]
         disappear_peds = self.peds[self.frame-1].difference(futu_peds)
         #print(disappear_peds)
         for ped in disappear_peds:
            if self.leave_map[ped] == 0:
               for i in range(k):
                  if ped in self.peds[self.frame+16+k]:
                     break
               else:
                  if self.is_at_boundary(ped, self.frame):
                     self.leave_map[ped] = 16
                  #print(ped)

      if self.frame < self.video_len:
         #for ped in self.peds[self.frame]:
         for row in self.curr_ped_arr:
            leave_state = self.leave_map[int(row[0])]
            if leave_state > 0:
               if leave_state % 2 == 0:
                  self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] =  \
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] * 0.5
                  self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] =  \
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] * 0.5 + 255 * 0.5
                  self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] =  \
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] * 0.5
               self.leave_map[int(row[0])] = leave_state - 1
      else:
         for i in range(1, self.total_peds+1):
            self.leave_map[i] = 0

   def ped_enter(self):
      
      if self.frame < self.video_len and self.frame > 3:
         
         futu_peds = self.peds[self.frame]
         occur_peds = futu_peds.difference(self.peds[self.frame-1])

         for ped in occur_peds:
            
            if self.is_at_boundary(ped, self.frame+1):
               self.enter_map[ped] = 16
               #print(ped)
         for row in self.curr_ped_arr:
            enter_state = self.enter_map[int(row[0])]
            if enter_state > 0:
               if enter_state % 2 == 0:   
                  self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] =  \
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] * 0.5 + 255 * 0.5
                  self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] =  \
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] * 0.5 
                  self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] =  \
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] * 0.5
                  
               self.enter_map[int(row[0])] = enter_state - 1
      else:
         for i in range(1, self.total_peds+1):
            self.enter_map[i] = 0

   def divide_group(self):
      self.group_label = [{i:i for i in range(1, self.total_peds+1)} for _ in range(self.video_len)]
      self.group_member = [{i:{i} for i in range(1, self.total_peds+1)} for _ in range(self.video_len)]

      for f in range(2,self.video_len):
         peds_list = list(self.peds[f-1])
         for i in range(len(peds_list)-1):
            ped1 = peds_list[i]
            for j in range(i+1, len(peds_list)):      
               ped2 = peds_list[j]
               if self.group_label[f-1][ped1] != self.group_label[f-1][ped2]:
                  # stay more than one frame
                  if self.is_close(ped1, ped2, f) and (self.is_close(ped1, ped2, f+1) or self.is_close(ped1, ped2, f-1)):
                     #if f == 230:
                     #   print(f'{ped1} {ped2} {f}')
                     group1 = self.group_label[f-1][ped1]
                     group2 = self.group_label[f-1][ped2]
                     for mem in self.group_member[f-1][group2]:
                        self.group_label[f-1][mem] = group1
                     self.group_label[f-1][ped2] = group1
                     self.group_member[f-1][group1].update(self.group_member[f-1][group2])
                     self.group_member[f-1][group2] = {group2}

      for f in range(2, self.video_len):
         for ped in self.peds[f-1]:
            group = self.group_member[f-1][ped]
            if len(group) > 1:
               group_t = tuple(sorted(group))
               if group_t in self.group_state:
                  self.group_state[group_t].append(f)
               else:
                  self.group_state[group_t] = [f]

      for g in self.group_state.copy():
         group_len = len(self.group_state[g])
         if group_len == 1:
            self.group_state.pop(g)
         else:
            temp = self.group_state[g].copy()
            for m in range(group_len-1, -1, -1):
               if m == group_len-1 and temp[m] != temp[m-1]+1:
                  self.group_state[g].pop(m)
               elif m == 0 and temp[m] != temp[m+1]-1:
                  self.group_state[g].pop(m)
               elif m > 0 and m < group_len-1 and temp[m] != temp[m+1]-1 and temp[m] != temp[m-1]+1:
                  self.group_state[g].pop(m)
            if not self.group_state[g]:
               self.group_state.pop(g)
         
   def group_form_and_dest(self):
      #form_and_dest[frame-1] = [{ped:intensity}, [(formated group)], {(destructing group):intensity}, [destruction peds]]#
      self.form_and_dest = [[{},[],{},[]] for _ in range(self.video_len)]

      for group, frame_list in self.group_state.items():
         length = len(frame_list)
         group_slice = []
         i = 0
         while i < length-1:
            for j in range(i, length-1):
               if frame_list[j] + 20 < frame_list[j+1]:
                  group_slice.append(frame_list[i:j+1])
                  i = j+1
                  break 
            else:
               group_slice.append(frame_list[i:j+2])
               break
         self.group_state[group] = group_slice

      for group, f_slice in self.group_state.items():
         for p in f_slice:
            l = len(p)
            if l < 4:
               continue
            if p[l-1] > self.video_len-3:
               continue

            disapp_peds = set()
            for ped in group:
               if ped not in self.peds[p[-1]]:
                  disapp_peds.add(ped)
            exit_peds = set(group).difference(disapp_peds)
            if len(exit_peds) < 2:
               continue

            follow_group = []
            for g in self.group_member[p[-1]].values():
               if len(g) > 1:
                  follow_group.append(g)

            for g in follow_group:
               #if set(group).issubset(g):
               if exit_peds.issubset(g):
                  break
            else:
               for i in range(l-min(12, l),l):
                  self.form_and_dest[p[i]-1][2][group] = (l-i)*0.6/min(12,l) + 0.2
               for i in range(min(14, self.video_len - p[l-1])):
                  for ped in group:
                     self.form_and_dest[p[l-1]+i][3].append(ped)

      for group, f_slice in self.group_state.items():
         for p in f_slice:
            l = len(p)
            if l < 4:
               continue   
            if p[0] < 3:
               continue

            ocurr_peds = set()
            for ped in group:
               if ped not in self.peds[p[0]-2]:
                  ocurr_peds.add(ped)
            exit_peds = set(group).difference(ocurr_peds)
            if len(exit_peds) < 2:
               continue

            pre_group = []
            for g in self.group_member[p[0]-2].values():
               if len(g) > 1:
                  pre_group.append(g)
            
            for g in pre_group:
               if exit_peds.issubset(g):
                  break
            else:
               for i in range(min(10, l)):
                  if group not in self.form_and_dest[p[i]-1][2]:
                     self.form_and_dest[p[i]-1][1].append(group)
               for i in range(1, min(17,p[0])):
                  for ped in group:
                     self.form_and_dest[p[0]-i-1][0][ped] = 0.6 - i*0.6/min(17,(p[0]-1))
                      

   def group_visual(self):
      for ped in self.peds[self.frame-1]:
         if self.enter_map[ped] > 0:
            break
         else:
            if ped in self.form_and_dest[self.frame-1][0]:
               a = self.form_and_dest[self.frame-1][0][ped]
               for row in self.curr_ped_arr:
                  if int(row[0]) == ped:
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] =  \
                        self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] * (1-a) + 255 * a
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] =  \
                        self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] * (1-a)
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] =  \
                        self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] * (1-a) + 255 * a

      for group in self.form_and_dest[self.frame-1][1]:
         for ped in group:
            if self.enter_map[ped] > 0:
               break
         else:
            x1, y1, x2, y2 = 1920, 1080, 0, 0
            for row in self.curr_ped_arr:
               if int(row[0]) in group:
                  if int(row[1]) < x1:
                     x1 = int(row[1])
                  if int(row[2]) < y1:
                     y1 = int(row[2])
                  if int(row[3]) > x2:
                     x2 = int(row[3])
                  if int(row[4]) > y2:
                     y2 = int(row[4])
            self.img[y1:y2, x1:x2, 0] =  self.img[y1:y2, x1:x2, 0] * 0.4 + 255 * 0.6
            self.img[y1:y2, x1:x2, 1] =  self.img[y1:y2, x1:x2, 1] * 0.4
            self.img[y1:y2, x1:x2, 2] =  self.img[y1:y2, x1:x2, 2] * 0.4 + 255 * 0.6

      for group,a in self.form_and_dest[self.frame-1][2].items():
         for ped in group:
            if self.leave_map[ped] > 0:
               break
         else:
            x1, y1, x2, y2 = 1920, 1080, 0, 0
            for row in self.curr_ped_arr:
               if int(row[0]) in group:
                  if int(row[1]) < x1:
                     x1 = int(row[1])
                  if int(row[2]) < y1:
                     y1 = int(row[2])
                  if int(row[3]) > x2:
                     x2 = int(row[3])
                  if int(row[4]) > y2:
                     y2 = int(row[4])
            self.img[y1:y2, x1:x2, 0] =  self.img[y1:y2, x1:x2, 0] * (1-a)
            self.img[y1:y2, x1:x2, 1] =  self.img[y1:y2, x1:x2, 1] * (1-a)
            self.img[y1:y2, x1:x2, 2] =  self.img[y1:y2, x1:x2, 2] * (1-a) + 255 * a

      for ped in self.peds[self.frame-1]:
         if self.leave_map[ped] > 0:
            break
         else:
            if ped in self.form_and_dest[self.frame-1][3]:
               for row in self.curr_ped_arr:
                  if int(row[0]) == ped:
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] =  \
                        self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 0] * 0.8
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] =  \
                        self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 1] * 0.8
                     self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] =  \
                        self.img[int(row[2]):int(row[4]), int(row[1]):int(row[3]), 2] * 0.8 + 255 * 0.2
          


   def is_close(self, ped1, ped2, frame_num):
      ped1_coord = None
      ped2_coord = None
      r1, r2 = 0.3, 0.5
      #k1, k2, k3, k4 = 4,1,2,0.2

      for row in np.array(self.bounding_boxes[frame_num-1].split(',')[1:]).reshape(-1,5):
         if int(row[0]) == ped1:
            ped1_coord = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
         if int(row[0]) == ped2:
            ped2_coord = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))

      if ped1_coord and ped2_coord:
         ped1_centre = ((ped1_coord[0]+ped1_coord[2])//2, (ped1_coord[1]+ped1_coord[3])//2)
         ped2_centre = ((ped2_coord[0]+ped2_coord[2])//2, (ped2_coord[1]+ped2_coord[3])//2)

         ped1_w = ped1_coord[2] - ped1_coord[0]
         ped1_h = ped1_coord[3] - ped1_coord[1]

         ped2_w = ped2_coord[2] - ped2_coord[0]
         ped2_h = ped2_coord[3] - ped2_coord[1]

         max_h = max(ped1_h, ped2_h)
         d_x = abs(ped2_centre[0]-ped1_centre[0])
         d_y = abs(ped2_centre[1]-ped2_centre[1])
         d_s = abs(ped2_coord[3]-ped1_coord[3])
         d = min(abs(ped1_w-ped2_w), abs(ped1_h-ped2_h))

         if d_s < r1 * max_h:
            if d_x < r2 * max_h:
               return True
         else:
            pass
           # if k1*d*d + k2*x*x + k3*y*y < max_h*max_h*k4:
           #    return True
      return False

# data_image_path = './train_boxes_numbered/0002',data_image_path = './rnn/train02_trained_weights_5epochs'
#PedTrackWin(data_image_path ='./train/STEP-ICCV21-02' )
if __name__ == '__main__':
   #PedTrackWin()
   #PedTrackWin(data_image_path = './A_test01/Image', box_path = 'A_test01.txt')
   PedTrackWin(data_image_path = './A_test01/ImageSTEP07', box_path = './A_test01/test07.txt')
   #PedTrackWin(data_image_path = './m_test/Track_07', box_path = './m_test/test_07.txt')
   
#,box_path = './rnn/train_02.txt'
