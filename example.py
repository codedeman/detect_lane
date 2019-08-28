import cv2
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
result  = np.random.randint(0,2,(25,1)).astype(np.float32)
red = trainData[result.ravel()==1]
blue = trainData[result.ravel()==0]

print('blue color',blue)
newMember  = np.random.randint(0,2,(1,2)).astype(np.float);
plt.scatter(red[:,0],red[:,1],100,'r','s')
plt.scatter(blue[:,0],blue[:,1],100,'b','^')
plt.scatter(newMember[:,0],newMember[:,1],100, 'g','o')

newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
# knn = cv2.ml.KNeareast_create()

# knn = cv2.ml.KNearest_create()
# knn.train(trainData,0, result)
# results = knn.findNearest(newMember, 3)
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, result)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 2)

plt.show()


# print(result)
#
# print('train data',trainData)





#
# import pygame
# pygame.init()
#
# win = pygame.display.set_mode((500,500))
#
# pygame.display.set_caption("First game")
#
# walkRight = [pygame.image.load('R1.png'), pygame.image.load('R2.png'), pygame.image.load('R3.png'), pygame.image.load('R4.png'), pygame.image.load('R5.png'), pygame.image.load('R6.png'), pygame.image.load('R7.png'), pygame.image.load('R8.png'), pygame.image.load('R9.png')]
# walkLeft = [pygame.image.load('L1.png'), pygame.image.load('L2.png'), pygame.image.load('L3.png'), pygame.image.load('L4.png'), pygame.image.load('L5.png'), pygame.image.load('L6.png'), pygame.image.load('L7.png'), pygame.image.load('L8.png'), pygame.image.load('L9.png')]
# bg = pygame.image.load('bg.jpg')
# char = pygame.image.load('standing.png')
#
# clock = pygame.time.Clock()
#
# class player(object):
#     def __init__(self,x,y,width,height):
#         self.x =  x
#         self.y = y
#         self.width = width
#         self.height = height
#         self.vel = 5
#         self.isJump = False
#         self.jumpCount = 10
#         self.left = False
#         self.right =  False
#         self.walkCount =0
#
#
#     def draw(self, win):
#         if self.walkCount + 1 >= 27:
#             self.walkCount = 0
#
#         if self.left:
#             win.blit(walkLeft[self.walkCount//3], (self.x,self.y))
#             self.walkCount += 1
#         elif self.right:
#             win.blit(walkRight[self.walkCount//3], (self.x,self.y))
#             self.walkCount +=1
#         else:
#             win.blit(char, (self.x,self.y))
#
# def redrawGameWindow():
#     win.blit(bg, (0, 0))
#     man.draw(win)
#
#     pygame.display.update()
#
# screenWidth = 500
# x = 50
# y = 425
#
# width  = 40
# height = 40
# vel: int = 5
#
# isJump =  False
# jumpCount = 10
#
# left = False
# right =  False
# walkCount = 0
#
#
#
# run = True
# while run:
#     pygame.time.delay(100)
#
#     for event  in pygame.event.get():
#         if event.type == pygame.QUIT:
#             run = False
#
#     keys = pygame.key.get_pressed()
#     #
#     #
#     if keys[pygame.K_LEFT] and x > vel:
#         x -= vel
#     if keys[pygame.K_RIGHT] and x< 500 - width - vel :
#         x += vel
#     if not(isJump):
#         if keys[pygame.K_UP] and y > vel:
#             y -= vel
#         if keys[pygame.K_DOWN] and y >500 - height - vel:
#             y += vel
#
#         if keys[pygame.K_SPACE]:
#             isJump =  True
#     else:
#          if jumpCount > -10:
#             neg = 1
#             if jumpCount < 0:
#                 neg = -1
#             y -= (jumpCount ** 2) * 0.5*neg
#             jumpCount -= 1
#
#          else:
#             isJump = False
#             jumpCount = 10
#
#
#
#
#
#     win.fill((0,0,0))
#     pygame.draw.rect(win,(255,0,0),(x,y,width,height))
#     pygame.display.update()
#
#
# pygame.quit()

# import numpy as np
# from PIL import Image
# img = Image.open('lena.png')
# arr = np.array(img) # 640x480x4 array
# print(arr)
# # arr[20, 30] # 4-vector, just like above