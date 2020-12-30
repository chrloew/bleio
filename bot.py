import discord
import os
import time

#from dotenv import load_dotenv



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
import io 
import urllib, base64 
from random import randint
import random

import asyncio

from boto.s3.connection import S3Connection

client = discord.Client()
#load_dotenv('.env')

reply_messages = [
	"Hmm, interessant! ",
	"Da habe ich mich selber übertroffen, dieser Zufall, so zufällig!",
	"Klare Sache, ist doch offensichtlich, man erkennt jedes einzelne Stichwort.",
	"Phiew, ich meine ja, die Stichworte waren kinda weird, aber ich habe das beste draus gemacht *uff* lol",
	"Naja, schaut komisch aus, aber bringt sicher Glück 🎩🐷"]

bleios_count = 0

@client.event
async def on_ready():
	print('We have logged in as {0.user}'.format(client))

	await client.change_presence(activity=discord.Game(name="!blei", type=1))

	#text_channel_list = []
	#for guild in client.guilds:
#		for channel in guild.text_channels:
#			text_channel_list.append(channel)
#	print(text_channel_list[0].id)
#	channel = client.get_channel(text_channel_list[-1].id)
#	await channel.send('hello')

@client.event
async def on_message(message):
	global bleios_count
	if message.author == client.user:
		return

	if message.content.startswith('$hello'):
		await message.channel.send('Hello!')
		
	if message.content.startswith('!blei'):
		await message.add_reaction('🎉')
		await message.channel.send(message.author.name+' schmilzt 🔥 das Blei 🪨 im Löffel 🥄 .. 🤵 .. ')
		bleio_filename = 'bleio_'+str(message.id)+'.png'
		bleio(bleio_filename)
		#time.sleep(randint(1,4))
		asyncwait(randint(7,10))
		#await message.channel.send('*pschhhht*')
		await message.channel.send('Uuuund.. _splash_ 💨 hier zu bewundern ist das Werk von '+ message.author.name+'!')
		await message.channel.send(file=discord.File(bleio_filename))
		#await message.channel.send('Hmm interessant!')
		await message.channel.send(random.choice(reply_messages))
		bleios_count += 1
		print('Created Led Pouring #'+str(bleios_count)+' for '+message.author.name+'.')
		os.remove(bleio_filename)




@client.event
async def on_reaction_add(reaction, user):
		"""Event handler for when reactions are added on the help message."""
		# ensure it was the relevant session message
		#if reaction.message.id != self.message.id:
		#    return

		# ensure it was the session author who reacted
		
		#if user.id != reaction.message.author.id:
		#    return

		#emoji = str(reaction.emoji)
		#await reaction.message.channel.send(emoji)


async def asyncwait(time):
    await asyncio.sleep(time)
    



bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
	N = len(points)
	t = np.linspace(0, 1, num=num)
	curve = np.zeros((num, 2))
	for i in range(N):
		curve += np.outer(bernstein(N - 1, i, t), points[i])
	return curve

class Segment():
	def __init__(self, p1, p2, angle1, angle2, **kw):
		self.p1 = p1; self.p2 = p2
		self.angle1 = angle1; self.angle2 = angle2
		self.numpoints = kw.get("numpoints", 100)
		r = kw.get("r", 0.3)
		d = np.sqrt(np.sum((self.p2-self.p1)**2))
		self.r = r*d
		self.p = np.zeros((4,2))
		self.p[0,:] = self.p1[:]
		self.p[3,:] = self.p2[:]
		self.calc_intermediate_points(self.r)

	def calc_intermediate_points(self,r):
		self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
									self.r*np.sin(self.angle1)])
		self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
									self.r*np.sin(self.angle2+np.pi)])
		self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
	segments = []
	for i in range(len(points)-1):
		seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
		segments.append(seg)
	curve = np.concatenate([s.curve for s in segments])
	return segments, curve

def ccw_sort(p):
	d = p-np.mean(p,axis=0)
	s = np.arctan2(d[:,0], d[:,1])
	return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
	""" given an array of points *a*, create a curve through
	those points. 
	*rad* is a number between 0 and 1 to steer the distance of
		  control points.
	*edgy* is a parameter which controls how "edgy" the curve is,
		   edgy=0 is smoothest."""
	p = np.arctan(edgy)/np.pi+.5
	a = ccw_sort(a)
	a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
	d = np.diff(a, axis=0)
	ang = np.arctan2(d[:,1],d[:,0])
	f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
	ang = f(ang)
	ang1 = ang
	ang2 = np.roll(ang,1)
	ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
	ang = np.append(ang, [ang[0]])
	a = np.append(a, np.atleast_2d(ang).T, axis=1)
	s, c = get_curve(a, r=rad, method="var")
	x,y = c.T
	return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
	""" create n random points in the unit square, which are *mindst*
	apart, then scale them."""
	mindst = mindst or .7/n
	a = np.random.rand(n,2)
	d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
	if np.all(d >= mindst) or rec>=200:
		return a*scale
	else:
		return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

def bleio(filename):
	fig, ax = plt.subplots()
	ax.set_aspect("equal")
	#ax.set_facecolor((1.0, 0.47, 0.42))


	fig.patch.set_facecolor('#36393E')
	fig.patch.set_alpha(0.7)

	ax.patch.set_facecolor('#36393E')
	ax.patch.set_alpha(0.5)

	positions=np.array([[0,0], [0,0.5], [0,1], [1,0], [1,0.5], [1,1], [0.5,0], [0.5,0.5], [0.5,1]])
	random_coords=np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8],randint(1,8), replace=False)
	#print(random_coords)
	for c2 in random_coords:
	#for c in np.array([[0,0], [0,0.5], [0,1], [1,0], [1,0.5], [1,1], [0.5,0], [0.5,0.5], [0.5,1]]):
	#for c in np.array([[0,0]]):
	

		random_rad = randint(2,7)/10
		rad = random_rad
		#rad = 0.2

		random_edgy = randint(10,100)/100
		edgy=random_edgy
		#edgy = 0.05

		
		c = positions[c2]
		# random offset
		c[0] = c[0] + randint(0,20)/100
		c[1] = c[1] + randint(0,20)/100
		
		a = get_random_points(n=randint(6,20), scale=0.4) + c
		x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
		color_theme=[randint(0,235)/235, randint(0,235)/235, randint(0,235)/235]
		ax.fill(x,y, color=color_theme)
		plt.plot(x,y, color=color_theme)

	plt.axis('off')
	plt.savefig(filename, bbox_inches='tight')

	#plt.show()
	
	return 1

client.run(os.environ['BOT_TOKEN'])