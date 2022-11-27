# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:55:29 2022

@author: nelso

#final project fractiles
"""



import numpy as np
import matplotlib.pyplot as plt
import array as array
import random as r
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# p1 = []
# p2 = []
# p3 = []
# p4
# p5
# p6
# points=[[],[]]


"""
its going to draw the shape

then its gonna roll a dice 
then starting at the first dot its gonna draw a dot 2/3 from the dice roll

"""
numberofnewpoints = 1
points = (())
diceroll = []
#n=0
numofsides = 0

showdata = 0
style2 = 1
printout = 0
test = 0
showlines= 0
v1 = 1

cmap = []


def addpointv3(p1,p2,n):
    
    #rearranges the point so that one of them is bigger than the other for easier math
    my = (p1[1] - p2[1])
    mx = (p1[0] - p2[0])
    
    bp=p1
    sp=p2
    
    #finds which point has the bigger xvalue and yvalue
    bx,sx=0; by,sy=0
    if p1[0]==p2[0]:
        print()
    elif p1[0] > p2[0]:
        bx = p1[0]
        sx = p2[0]
    elif p1[0] < p2[0]:
        bx = p2[0]
        sx = p1[0]
        
    if p1[1]==p2[1]:
        print()
    elif p1[1] > p2[1]:
        by = p1[1]
        sy = p2[1]
    elif p1[1] < p2[1]:
        by = p2[1]
        sy = p1[1]
    #end
    
    # #### data about the points
    # my = (p1[1] - p2[1])
    # mx = (p1[0] - p2[0])
    
    # h= d = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
    # x23 = (1/3*(h*np.cos(findtheta(bx, sx))))
    # y23 = (1/3*(h*np.sin(findtheta(p1, sy))))
    # ####
    
    
    if bp[1]==sp[1]:
        print("\tveritical line")
    
    elif bp[0]==sp[0]:
        print("\thorizontal line")
    

def addpoint(p1, p2,n):
    print(f"\n ({p1[0]:.2f},{p1[1]:.2f}) : ({p2[0]:.2f},{p2[1]:.2f})")
    #xval =  ( ( 2 *( p1[0] + p2[0] ))/3 )               #p1[0] + ( ( 2 * np.abs(( p1[0] + p2[0] )))/3 )     #( ( p1[0] + p2[0] )/2 )
    #yval =  ( ( 2 *( p1[1] + p2[1] ))/3 )               #p1[1] + ( ( 2 * np.abs(( p1[1] + p2[1] )))/3 )     #( ( p1[1] + p2[1] )/2 )
    
    #xval =  ( ( p1[0] + p2[0] )/2 )
    #yval =  ( ( p1[1] + p2[1] )/2 )
    
    
    my = (p1[1] - p2[1])
    mx = (p1[0] - p2[0])
    print("my is {my}, mx is {mx}")
    if(mx == 0):
        print("skipped")
    else:
        m = (p1[1] - p2[1])/(p1[0] - p2[0])
        # print(m)
    b = 1
    
    if my >= 0 and mx >=0:
        xval =  ( ( p1[0] + p2[0] )/2 ) + b
        yval =  ( ( p1[1] + p2[1] )/2 ) + b
    
    elif mx<=0 and my <=0:
        xval =  ( ( p1[0] + p2[0] )/2 ) - b
        yval =  ( ( p1[1] + p2[1] )/2 ) - b
    
    elif mx<=0 and my >= 0:
        xval =  ( ( p1[0] + p2[0] )/2 ) + b
        yval =  ( ( p1[1] + p2[1] )/2 ) - b
        
    elif mx>=0 and my <= 0:
        xval =  ( ( p1[0] + p2[0] )/2 ) + b
        yval =  ( ( p1[1] + p2[1] )/2 ) - b
    
    print (f" xval: {xval: .2f} , yval:{yval: .2f}")
    #points.add(xval,yval)
    plt.scatter(xval,yval)
    plt.Annotation("fuck", (xval,yval))









def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) 
                  for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") 
                           for val in item]) 
            for item in rgb_colors]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"fuck":[RGB_to_hex(RGB) for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

from numpy import random as rnd

def rand_hex_color(num=1):
  ''' Generate random hex colors, default is one,
      returning a string. If num is greater than
      1, an array of strings is returned. '''
  colors = [
    RGB_to_hex([x*255 for x in rnd.rand(3)])
    for i in range(num)
  ]
  if num == 1:
    return colors[0]
  else:
    return colors


def polylinear_gradient(colors, n):
  ''' returns a list of colors forming linear gradients between
      all sequential pairs of colors. "n" specifies the total
      number of desired output colors '''
  # The number of colors per individual linear gradient
  n_out = int(float(n) / (len(colors) - 1))
  # returns dictionary defined by color_dict()
  gradient_dict = linear_gradient(colors[0], colors[1], n_out)

  if len(colors) > 1:
    for col in range(1, len(colors) - 1):
      next = linear_gradient(colors[col], colors[col+1], n_out)
      for k in ("hex", "r", "g", "b"):
        # Exclude first point to avoid duplicates
        gradient_dict[k] += next[k][1:]

  return gradient_dict

def jigglephysics(n):
    tempthing = linear_gradient("#8c03fc", "#fc03be", numberofnewpoints)
    ccolor = tempthing['fuck']
    return ccolor[n-1]






def whocloser(cp,newpoint):
    distances=[]
    nep = newpoint
    for x in range(6):
        d = np.sqrt( (cp[x][0] - nep[0])**2 + (cp[x][1] - nep[1])**2 )
        distances.append(d)
    for x in range(len(distances)):
        closest = distances[0]
        if closest > distances[x]:
            closest = distances[x]
    return closest

def findtheta(p1,p2):
    my = (p2[1] - p1[1])#my = (p1[1] - p2[1])
    mx = (p2[0] - p1[0])#mx = (p1[0] - p2[0])
    
    theta = np.arctan(my/mx)
    
    return theta
    
def addpointv2(p1, p2,n):
    print(f"\n{n}\n ({p1[0]:.2f},{p1[1]:.2f}) : ({p2[0]:.2f},{p2[1]:.2f})")
    
    #xval =  ( ( 2 *( p1[0] + p2[0] ))/3 )               #p1[0] + ( ( 2 * np.abs(( p1[0] + p2[0] )))/3 )     #( ( p1[0] + p2[0] )/2 )
    #yval =  ( ( 2 *( p1[1] + p2[1] ))/3 )               #p1[1] + ( ( 2 * np.abs(( p1[1] + p2[1] )))/3 )     #( ( p1[1] + p2[1] )/2 )
    
    # if p1[0] > p2[0]:
    #     print()
    # elif p1[0] < p2[0]:
    #     t=p2[0]
    #     p2[0]=p1[0]
    #     p1[0]=t
    # if p1[1] > p2[1]:
    #     print()
    # elif p1[1] < p2[1]:
    #     t=p2[1]
    #     p2[1]=p1[1]
    #     p1[1]=t
    
    xval =  ( ( p1[0] + p2[0] )/2 )
    yval =  ( ( p1[1] + p2[1] )/2 )
    if showdata == True: plt.plot(xval, yval,"k.")
    if showdata == True: plt.text(xval,yval,f"{n}mid",fontsize =8)
    
    my = (p2[1] - p1[1])#my = (p1[1] - p2[1])
    mx = (p2[0] - p1[0])#mx = (p1[0] - p2[0])
    
    d = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
    
    if showdata == True:print(f"my is {my:.2f}, mx is {mx:.2f}, d is {d:.2f}")
    
    if mx == 0.0:
        print("skipped")
    else:
        m = my/mx#m = (p1[1] - p2[1])/(p1[0] - p2[0])
        print(f"the slope is: {m:.2f}")
        # print(m)
    b = 1
    
    h=d
    
    #x23 = ((10*np.sqrt(3))/3)
    #y23 = (10/3)
    ch = h * 1/6
    if mx!=0 and my !=0:
        x23 = ((ch*np.cos(findtheta(p1, p2))))#x23 = (1/3*(h*np.cos(findtheta(p1, p2))))
        y23 = ((ch*np.sin(findtheta(p1, p2))))#y23 = (1/3*(h*np.sin(findtheta(p1, p2))))
        print(f"x23 ={x23:.2f} , y23 = {y23:.2f}")
    
    
    if my > 0 and mx >0:
        print("method 1")
        xval = xval + x23
        yval = yval + y23
        print(f"!!!!!{x23}, {y23}, {d}")
        
    elif mx<0 and my <0:
        print("method 2")
        xval = p1[0] - x23
        yval = p1[1] - y23
        print("testing")
    
    elif mx<0 and my >0:
        print("method 3")
        xval = xval + x23#xval = p1[0] + (2/3*(h*np.cos(30)))
        yval = yval + y23#yval = p1[1] - (2/3*(h*np.sin(30)))
        
    elif mx>0 and my < 0:
        print("method 4")
        xval = xval + x23
        yval = yval + y23
        print(f"4{x23}, {y23}, {d}")
        
    elif mx ==0 and my !=0:
        print("method 5")
        xval = xval 
        print("x==0 ", yval,"+" ,(ch*np.sin(30)))
        yval = yval + (ch*np.sin(30))#y23#yval = yval + (2/3*(h*np.sin(30)))
        
        
    elif mx !=0 and my ==0:
        print("method 6")
        print("y==0", xval)
        xval = xval + (ch*np.sin(30))#(2/3*(h*np.cos(30)))
        yval = yval 
        
    
    print (f" xval: {xval: .2f} , yval:{yval: .2f}")
    #points.add(xval,yval)
    
    #THIS IS THE NEW POINTS
    plt.scatter(xval,yval, s = 4,c=1)
    
    #THIS PLOTS THE END POINT OF EACH OF THEM
    plt.scatter(p1[0],p1[1], s = 4, c=1)
    
    #plt.annotation("h", (xval,yval))
    if showdata == True: plt.text(xval,yval,f"{n}[{xval: .2f},{yval: .2f}]", fontsize = 7)
    plt.text(xval,yval,f"{n}", fontsize = 7)
    #plt.arrow(xval, yval, p1[0], p1[1])
    #plt.arrow(xval, yval, p2[0], p2[1])
    xs = [p1[0],xval,p2[0]]
    ys = [p1[1],yval,p2[1]]
    
    #THIS SHOWS THE LINES FROM THE START AND FINISH OF THE NEW POINTS
    if showdata == True: plt.plot(xs, ys, '1', linestyle="--", lw =1)#you can change the lw to make it thicker
    
    if style2 == True:
        newpoint = [xval,yval]
        return newpoint









def addpointv4(p1, p2,n):
    print(f"\n{n}\n ({p1[0]:.2f},{p1[1]:.2f}) : ({p2[0]:.2f},{p2[1]:.2f})")
    
    #xval =  ( ( 2 *( p1[0] + p2[0] ))/3 )               #p1[0] + ( ( 2 * np.abs(( p1[0] + p2[0] )))/3 )     #( ( p1[0] + p2[0] )/2 )
    #yval =  ( ( 2 *( p1[1] + p2[1] ))/3 )               #p1[1] + ( ( 2 * np.abs(( p1[1] + p2[1] )))/3 )     #( ( p1[1] + p2[1] )/2 )
    
    # if p1[0] > p2[0]:
    #     print()
    # elif p1[0] < p2[0]:
    #     t=p2[0]
    #     p2[0]=p1[0]
    #     p1[0]=t
    # if p1[1] > p2[1]:
    #     print()
    # elif p1[1] < p2[1]:
    #     t=p2[1]
    #     p2[1]=p1[1]
    #     p1[1]=t
    
    xval =  ( ( p1[0] + p2[0] )/2 )
    yval =  ( ( p1[1] + p2[1] )/2 )
    if showdata == True: plt.plot(xval, yval,"k.")
    if showdata == True: plt.text(xval,yval,f"{n}mid",fontsize =8)
    
    my = (p2[1] - p1[1])#my = (p1[1] - p2[1])
    mx = (p2[0] - p1[0])#mx = (p1[0] - p2[0])
    
    d = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
    
    if showdata == True:print(f"my is {my:.2f}, mx is {mx:.2f}, d is {d:.2f}")
    
    if mx == 0.0:
        print("skipped")
    else:
        m = my/mx#m = (p1[1] - p2[1])/(p1[0] - p2[0])
        print(f"the slope is: {m:.2f}")
        # print(m)
    b = 1
    
    h=d
    
    #x23 = ((10*np.sqrt(3))/3)
    #y23 = (10/3)
    ch = h * 1/6
    
    
    if p1[0]!=p2[0] and p1[1]!=p2[1]: #mx!=0 and my !=0:
        
        x23 = ((ch*np.cos(findtheta(p1, p2))))#x23 = (1/3*(h*np.cos(findtheta(p1, p2))))
        y23 = ((ch*np.sin(findtheta(p1, p2))))#y23 = (1/3*(h*np.sin(findtheta(p1, p2))))
        print(f"x23 ={x23:.2f} , y23 = {y23:.2f}")
        
        if  mx > 0 and my > 0 :
            print("method 1")
            xval = xval + x23
            yval = yval + y23
            print(f"!!!!!{x23}, {y23}, {d}")
            
        elif mx<0 and my <0:
            print("method 2")
            xval = xval - x23
            yval = yval - y23
            print("testing")
        
        elif mx<0 and my >0:
            print("method 3")
            xval = xval - x23#xval = p1[0] + (2/3*(h*np.cos(30)))
            yval = yval - y23#yval = p1[1] - (2/3*(h*np.sin(30)))
            
        elif mx>0 and my < 0:
            print("method 4")
            xval = xval + x23
            yval = yval + y23
            print(f"4{x23}, {y23}, {d}")
    
    else:
        
        if mx ==0 and my > 0:
            print("method 5")
            xval = xval 
            print("x==0 ", yval,"+" ,(ch*np.sin(30)))
            yval = yval + ch#(ch*np.sin(30))#y23#yval = yval + (2/3*(h*np.sin(30)))
            
        elif mx ==0 and my < 0:
            print("method 6")
            print("y==0", xval)
            xval = xval #(2/3*(h*np.cos(30)))
            yval = yval - ch#(ch*np.sin(30))
            
        elif mx > 0 and my ==0:
            print("method 7")
            print("y==0  mid", xval)
            xval = xval + ch#(ch*np.cos(30))#(2/3*(h*np.cos(30)))
            yval = yval 
        
        elif mx < 0 and my ==0:
            print("method 8")
            print("y==0", xval)
            xval = xval - ch#(ch*np.cos(30))#(2/3*(h*np.cos(30)))
            yval = yval 
        
    
    print (f" xval: {xval: .2f} , yval:{yval: .2f}")
    #points.add(xval,yval)
    
    """THIS IS VERY IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    #THIS IS THE NEW POINTS
    plt.scatter(xval,yval, s = 1,c=jigglephysics(n-1),marker = '.')
    """THIS IS VERY IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    
    #THIS PLOTS THE END POINT OF EACH OF THEM
    plt.scatter(p1[0],p1[1], s =.5, c="black",marker = "x")#plt.scatter(p1[0],p1[1], s = 5, c="green",marker = "X",edgecolors="black")
    
    #plt.annotation("h", (xval,yval))
    if showdata == True: plt.text(xval,yval,f"{n}[{xval: .2f},{yval: .2f}]", fontsize = 7)
    
   #NUMBERS EACH AND EVERY NEWPOINT
    #plt.text(xval,yval,f"{n}", fontsize = 7)
    
    #plt.arrow(xval, yval, p1[0], p1[1])
    #plt.arrow(xval, yval, p2[0], p2[1])
    xs = [p1[0],xval,p2[0]]
    ys = [p1[1],yval,p2[1]]
    
    #THIS SHOWS THE LINES FROM THE START AND FINISH OF THE NEW POINTS
    if showdata == True: plt.plot(xs, ys, '1', linestyle="--", lw =1)#you can change the lw to make it thicker
    #, c= np.get_color_gradient("#ff0000", "#d300ff", len(numberofnewpoints))
    
    #MAKING THE COLOR SHIT WORK FOR THE LINES
    
    #tempthing = linear_gradient("#ff0000", "#03fc18", numberofnewpoints) ;  print(tempthing)
    #ccolor = tempthing['fuck'] ;  print(ccolor)
    #ccolor = np.array(ccolor) ; print(ccolor)#ccolor =["fuck"].values(); print(ccolor)#ccolor = cm.get_cmap('viridis',numberofnewpoints)#ccolor = get_color_gradient("#ff0000", "#d300ff", numberofnewpoints)
    #, color = ccolor.count(n)              color = ccolor[n-1]
    if showlines == True: plt.plot(xs, ys, color = jigglephysics(n),linestyle="solid", lw =1)#you can change the lw to make it thicker
    
    if style2 == True:
        newpoint = [xval,yval]
        return newpoint




def makeshape():
    
    sides = 6
    global numofsides
    numofsides = sides
    #angles = np.random.uniform(0,360,sides)
    angles = [0,60,120,180,240,300,360]
    print(angles)
    
    cornerpoints = np.zeros((sides,2))
    
    cushion = 10
    sidelength=20
    centerpoint = [sidelength + cushion, sidelength + cushion]
    print(f"the center is at {centerpoint}")
    
    #x = 4 + np.random.normal(0, 2, 24)
    #colors = np.random.uniform(15, 80, len(x))
    #fig,ax = plt.subplots()
    
    plt.scatter(centerpoint[0],centerpoint[1])#plt.plot(centerpoint)
    
    
    
    for x in range(0,sides):
        
        tempangle =angles[x]#tempangle = x * angles[x]
        print(tempangle)
        
        radangle = ((tempangle * np.pi)/180)
                
        adj = centerpoint[0] + (sidelength * np.cos(radangle)) 
        
        opp = centerpoint[1] + (sidelength * np.sin(radangle))
        
        cornerpoints[x][0] = adj        # cornerpoints[x] = [adj , opp]
        cornerpoints[x][1] = opp
        
        print(cornerpoints[x])
        cp = cornerpoints
        print(f"{cornerpoints[x][0]:.2f}")
        plt.scatter(cornerpoints[x][0],cornerpoints[x][1],color = "black")
        
        #UNCOMMENT THIS
        #plt.text(adj,opp,f"{x}({adj:.2f},{opp:.2f})")
        plt.text(adj,opp,f"{x}")
        
    #check the distances between the outer points
    for b in range(0,5):
        d = np.sqrt( (cp[b+1][0] - cp[b][0])**2 + (cp[b+1][1] - cp[b][1])**2 )
        print(f"{d} for points {b}, {b+1}")
    for b in range(0,6):
        d = np.sqrt( (cp[b][0] - centerpoint[0])**2 + (cp[b][1] - centerpoint[1])**2 )
        print(f"{d} for center and point {b}")
    
    # ## makes it so you can see the lines of the hexagon
    # xv = np.zeros((sides+1,2))
    # yv = np.zeros((sides+1,2))
    # for v in range(0,sides):
    #     xv[v] = cp[v][0]
    #     yv[v] = cp[v][1]
    # xv[sides] = cp[0][0]
    # yv[sides] = cp[0][1] 
    # plt.plot(xv, yv,"black" ,linestyle=":")
    # ## end
    #plt.grid()
    #plt.show()
    return cp

def ran():
    ran =r.randint(0, 5)
    return ran

def newroll():
    roll = ran()
    
    if roll == diceroll[len(diceroll)-1]:
        newroll = 0
        while newroll == roll:
            newroll = ran()
        roll = newroll
    return roll
    

def code(iteratios,cp):
    global numberofnewpoints 
    numberofnewpoints = iteratios
    global diceroll
    global cmap; 
    cmap = np.arange(0,numberofnewpoints,numberofnewpoints)
    #global n
   ###first method of ploting 
    if style2 == False and test == False:
        n = 0
        for x in range(0,(int)(iteratios)):
            for y in range(0,6):#for y in range(0,6)
                n+=1
                dice = r.randint(0, 5)#5
                if(y != dice):
                    p1 = start = cp[y]
                    p2 = end = cp[dice]
                    #print(f"p1 = {cp[y]:.2f} , p2 is {end:.2f}")
                    addpointv4(start, end,n)     # addpoint(cp[y],cp[dice])
   ##################end
    if style2 == True:
        n=0
        start = cp[0]    #OLD AND WRONG start = cp[r.randint(0, 5)]
        ran =r.randint(0, 5)
        end = cp[5]#end = cp[ran]
        diceroll.append(ran)
        newpoint = addpointv4(start, end, n)
        for x in range(0,(int)(iteratios)):
            
            n+=1
            
            if v1 == True:
                #version 1
                dice = r.randint(0, 5)#5 
                diceroll.append(dice)
            else:
                #version 2
                dice = newroll()
                diceroll.append(dice)
            
            p1 = start = newpoint
            p2 = end = cp[dice]
            #print(f"p1 = {cp[y]:.2f} , p2 is {end:.2f}")
            newpoint = addpointv4(start, end,n)     # addpoint(cp[y],cp[dice])
            
            whocloser(cp,newpoint)
        print(diceroll[:])
    ##################################################################################
    if test == True:
        for x in range(numofsides-1):
            start = cp[x]
            end = cp[x+1]
            addpointv4(start, end,x)
            
        start = cp[numofsides-1]
        end = cp[0]
        addpointv4(start, end,x);x+=1
        start = cp[1]
        end = cp[5]
        addpointv4(start, end,x)
            

def box():
    plt.plot(5,20)
    plt.plot(10,20)
    plt.plot(5,15)
    plt.plot(10,15)
    
    cp = np.zeros((4,2))
    cp[0,:]=[5,20]
    cp[1,:]=[10,20]
    cp[2,:]=[5,15]
    cp[3,:]=[10,15]
    x = [5, 10 , 5 , 10]
    y = [20,20,15,15]
    print(cp)
    
    plt.plot(x, y,"k.")
    #plt.scatter(cornerpoints[x][0],cornerpoints[x][1])
    return cp


def data(cp):
    print("\nData:")
    for b in range(0,5):
        d = np.sqrt( (cp[b+1][0] - cp[b][0])**2 + (cp[b+1][1] - cp[b][1])**2 )
        print(f"{d:.2f} for points {b}, {b+1}")



cp = makeshape()
#cp = box()

data(cp)
"""$$$$$$$$$$$$$$$$$$$$ for more iterations change the 800 to your desired number $$$$$$$$$$$$$$$$$$$"""
code(800, cp)
plt.grid()
plt.show()

        
        
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))
# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()  
"""
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        