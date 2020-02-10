#!/usr/bin/env python

import getch
import rospy
import time
from std_msgs.msg import Int8

keycheck_data = 0

def keycheck(msg):
	global keycheck_data
	keycheck_data = msg.data
	#print("received: ",keycheck_data)


def keys():
	global keycheck_data	
	rospy.Rate(100)
	while not rospy.is_shutdown():
		try:
			key_input = ord(getch.getch())# this is used to convert the keypress event in the keyboard or joypad , joystick to a ord value
			key_pub.publish(key_input)
			#print("input = ",key_input)
			time.sleep(0.03)
			if keycheck_data != key_input:
				key_pub.publish(key_input)
		except Exception:
			pass


if __name__=='__main__':
	try:
		key_pub = rospy.Publisher('key_input',Int8,queue_size=10) # "key" is the publisher name
		keychk_sub = rospy.Subscriber("key_check",Int8,keycheck,queue_size=10)
		rospy.init_node('Keyboard_Input',anonymous=True)
		keys()
	except rospy.ROSInterruptException:
		pass