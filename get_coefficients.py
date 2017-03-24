# file: get_coefficients.py
# -*- coding: utf-8 -*-
# author: xiaocen
# date: 2017.02.19
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from scipy import linalg

def gaussian(t, sigma):
	ret = np.exp(- t**2 / (2.0 * sigma**2))
	return ret

def h(k, c, s, w, b, sigma):
	ret = (c[0] * np.cos(w[0] * k / sigma) + s[0] * np.sin(w[0] * k / sigma)) * np.exp(- b[0] * k / sigma)
	for ci, si, wi, bi in zip(c[1:], s[1:], w[1:], b[1:]):
		ret += (ci * np.cos(wi * k / sigma) + si * np.sin(wi * k / sigma)) * np.exp(- bi * k / sigma)
	return ret	

def dh(k, c, s, w, b, sigma):
	n = c.shape[0]
	if type(k) == type(1.0):
		ret = np.zeros((4 * n, ), dtype = np.float32)
	else:
		ret = np.zeros((4 * n, k.shape[0]), dtype = np.float32)
	for i in range(n):
		hi = (c[i] * np.cos(w[i] * k / sigma) + s[i] * np.sin(w[i] * k / sigma)) * np.exp( - b[i] * k / sigma )
		ret[i]         = np.cos(w[i] * k / sigma) * np.exp( - b[i] * k / sigma )
		ret[i + n]     = np.sin(w[i] * k / sigma) * np.exp( - b[i] * k / sigma )
		ret[i + 2 * n] = (c[i] * np.sin(w[i] * k / sigma) * ( - k / sigma ) + s[i] * np.cos(w[i] * k / sigma) * ( k / sigma )) * np.exp( - b[i] * k / sigma ) 
		ret[i + 3 * n] = hi * ( - k / sigma )
	return ret

def hdh(k, c, s, w, b, sigma):
	n = c.shape[0]
	if type(k) == type(1.0):
		ret = np.zeros((4 * n, ), dtype = np.float32)
	else:
		ret = np.zeros((4 * n, k.shape[0]), dtype = np.float32)
	h = (c[0] * np.cos(w[0] * k / sigma) + s[0] * np.sin(w[0] * k / sigma)) * np.exp(- b[0] * k / sigma) 
	for i in range(n):
		hi = (c[i] * np.cos(w[i] * k / sigma) + s[i] * np.sin(w[i] * k / sigma)) * np.exp( - b[i] * k / sigma )
		if i > 0:
			h += hi
		ret[i]         = np.cos(w[i] * k / sigma) * np.exp( - b[i] * k / sigma )
		ret[i + n]     = np.sin(w[i] * k / sigma) * np.exp( - b[i] * k / sigma )
		ret[i + 2 * n] = (c[i] * np.sin(w[i] * k / sigma) * ( - k / sigma ) + s[i] * np.cos(w[i] * k / sigma) * ( k / sigma )) * np.exp( - b[i] * k / sigma ) 
		ret[i + 3 * n] = hi * ( - k / sigma )
	return h, ret

def gaussian_obj(x, k, sigma):
	n = int(x.shape[0] / 4)
	c = x[:n]
	s = x[n:2 * n]
	w = x[2 * n:3 * n]
	b = x[3 * n:]
	term = (h(k, c, s, w, b, sigma) - gaussian(k, sigma))**2
	return term.sum()

def gaussian_obj_prime(x, k, sigma):
	n = int(x.shape[0] / 4)
	c = x[:n]
	s = x[n:2 * n]
	w = x[2 * n:3 * n]
	b = x[3 * n:]
	hterm, dhterm = hdh(k, c, s, w, b, sigma)
	term = 2 * (hterm - gaussian(k, sigma))
	fprime = [x for x in map(lambda x : (term * x).sum(), dhterm)]
	fprime = np.array(fprime)
	return fprime

def getZerosPoles(N, order = 4, sigma = 100):
	N = 10 * sigma if N < 10 * sigma else N
	k = np.arange(0, N * (1 + 1.0 / 10000.0), N / 10000.0)
	x0 = np.zeros((2 * order, ), dtype = np.float32)				
#	x = optimize.fmin_bfgs(gaussian_obj, x0, gaussian_obj_prime, (k, sigma), maxiter = 100)
	x = optimize.fmin_bfgs(gaussian_obj, x0, None, (k, sigma), maxiter = 500)
	n = int(order / 2)
	c = x[:n]
	s = x[n:2 * n]
	w = x[2 * n:3 * n]
	b = x[3 * n:]
	return c, s, w, b

def getCoefficients(c, s, w, b, sigma):
	order = c.shape[0] * 2
	poles = []
	ai = []
	for ci, si, wi, bi in zip(c, s, w, b):
		poles.append(np.exp(- bi / sigma + 1j * wi / sigma))
		poles.append(np.exp(- bi / sigma - 1j * wi / sigma))
		ai.append(0.5 * (ci - 1j * si))
		ai.append(0.5 * (ci + 1j * si))
	poles = np.array(poles, dtype = np.complex)
	ai = np.array(ai, dtype = np.complex)
	for i, pole in enumerate(poles):
		for j, jpole in enumerate(poles):
			if not i == j:
				ai[i] *= 1.0 - jpole / pole 
	mat = np.zeros((order, order), dtype = np.complex)
	pow = np.arange(-1, -order -1, -1)
	for i, pole in enumerate(poles):
		mat[i] = pole**pow
	rhs = np.ones((order, ), dtype = np.complex)
	rhs = - 1 * rhs
	a = linalg.solve(mat, rhs)
	pow = np.arange(0, -order, -1)
	for i, pole in enumerate(poles):
		mat[i] = pole**pow
	b_plus = linalg.solve(mat, ai)
	return a.real, b_plus.real

def gaussianIIR(src, a, b_plus):
	radius = a.shape[0]
	b0 = b_plus[0]
	b_minus = [x for x in map(lambda x : x[1] - b0 * x[0], zip(a[:-1], b_plus[1:]))]
	b_minus.append(- b0 * a[-1]) 
	b_minus = np.array(b_minus, dtype = np.float32)
	print b_minus
	nSamples = src.shape[0]
	h_plus = np.zeros((nSamples, ), dtype = np.float32)
	for idx in range(nSamples):
		h_plus[idx] = b_plus[0] * src[idx]
		for k in range(1, radius + 1):
			xk = 0.0
			if k < radius: 
				xk = src[idx - k] if idx >= k else src[0]
			hk = h_plus[idx - k] if idx >= k else src[0]
			if k < radius:
				h_plus[idx] += b_plus[k] * xk 
			h_plus[idx] += - a[k - 1] * hk
	
	h_minus = np.zeros((nSamples, ), dtype = np.float32)
	for idx in range(nSamples - 1, -1, -1):
		for k in range(1, radius + 1):
			xk = src[idx + k] if idx <= nSamples - 1 - k else src[nSamples - 1]
			hk = h_minus[idx + k] if idx <= nSamples - 1 - k else src[nSamples - 1]
			h_minus[idx] += b_minus[k - 1] * xk - a[k - 1] * hk
	
	return h_plus + h_minus


if __name__ == "__main__":
	from struct import pack
#	f = open('coeff_tab_4order', 'wb')
#	dSigma = 0.01
#	sigmaMax = 100
#	sigma = np.arange(1.0/3.0, sigmaMax + dSigma, dSigma)
#	for s in sigma:
#		a, b_plus = getCoefficients(1000, 4, s)
#		a.tofile(f)
#		b_plus.tofile(f)
#	f.close()	
#
	c, s, w, b = getZerosPoles(1000, 4, 100)
	from sys import argv
	sigma_ = float(argv[1])
	print c, s, w, b
#	c = c * 100 / sigma_
#	s = s * 100 / sigma_

#	k = np.arange(0, 10 * sigma_, 0.1, dtype = np.float32)
#	g = gaussian(k, sigma_)
#	hh = h(k, c, s, w, b, sigma_)
#	print(((g - hh)**2).sum())
#
#	fig = plt.figure(figsize = (8, 8))
#	ax = fig.add_subplot(1, 1, 1)
#	ax.plot(k, g, 'r')
#	ax.plot(k, hh, 'b')
#	plt.show()
	src = np.zeros((1000, ), dtype = np.float32)
	src[300] = 0.5
	src[500] = 0.5
	src[800] = 0.5

	a, b_plus = getCoefficients(c, s, w, b, sigma_)
	b_plus /= np.sqrt(2.0 * np.pi) * sigma_
	print a, b_plus
#	b_plus /= np.sqrt(2 * np.pi) * sigma_
	dst = gaussianIIR(src, a, b_plus)
	t = np.arange(0, 1000, 1)
#	plt.plot(t, src, 'r')
	plt.plot(t, dst, 'b')
	plt.show()
