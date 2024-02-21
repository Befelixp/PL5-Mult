import matplotlib.pyplot as plt #3.1
import matplotlib.colors as clr #3.2
import numpy as np #decoder
import cv2 
import scipy.fftpack as fft

#-----Ex.2-----

#Encoder

def splitRGB(img): #separar matrizes
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B

def encoder(img):
    R, G, B= splitRGB(img)
    #Padding caso dimensão não seja multiplo de 32x32 (copia a ultima linha/coluna para preencher o espaço em falta)
    #4.1
    nl, nc= R.shape #n linhas e colunas
    nl_pad= 32-(nl%32)
    nc_pad= 32-(nc%32)
    R= np.pad(R, ((0, nl_pad), (0, nc_pad)), mode= 'edge') #edge= copia a ultima linha/coluna
    G= np.pad(G, ((0, nl_pad), (0, nc_pad)), mode= 'edge')
    B= np.pad(B, ((0, nl_pad), (0, nc_pad)), mode= 'edge')
    return R, G, B

#Decoder

def joinRGB(R, G, B):
    nl, nc= R.shape #n linhas e colunas
    imgRec= np.zeros((nl, nc, 3), dtype=np.uint8) #array inicializado a 0

    imgRec[:, :, 0] = R #img reconstruida no canal 0 para vermelho
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B

    return imgRec

def decoder(R,G,B, tamOriginal):
    imgRec= joinRGB(R, G, B)
    #4.2
    #remover padding
    imgRec= imgRec[:tamOriginal[0], :tamOriginal[1], :]
    return imgRec

def showImg(img, cmap= None, caption= ""):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(caption)

#5.1
def RGB_to_yCbCr(R, G, B):
    #converter para YCbCr
    #depois alterar pro do slide 46 M1.1 (com divisão)
    Y= 0.299*R + 0.587*G + 0.114*B
    #Cb= -0.168736*R - 0.331264*G + 0.5*B + 128
    Cb = ((B - Y) / 1.772) + 128
    #Cr= 0.5*R - 0.418688*G - 0.081312*B + 128
    Cr = ((R - Y) / 1.402) + 128
    return Y, Cb, Cr

def yCbCr_to_RGB(Y, Cb, Cr):
    #converter para RGB
    R= np.clip((Cr-128)*1.402 + Y, 0, 255) #clip para valores entre 0 e 255
    B= np.clip((Cb-128)*1.772 + Y, 0, 255)
    G= np.clip((Y - (0.299*R) - (0.114*B))/0.587, 0, 255)
    return R, G, B

# ---Ex 6---
def downsampling(Y,Cb, Cr, n):
    if n == 444:
        return Y, Cb, Cr
    if n == 422:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=1, interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=1, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_d, Cr_d
    if n == 420:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_d, Cr_d
    return print("valor de n errado!")

def upsampling(Y,Cb, Cr, n):
    if n == 422:
        Cb_u = cv2.resize(Cb,None, fx=2, fy=1, interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr,None, fx=2, fy=1, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_u, Cr_u
    if n == 420:
        Cb_u = cv2.resize(Cb,None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr,None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_u, Cr_u
    return print("valor de n errado!")

# ---- Ex 7 ----

#7.1.1
def dct(Y, Cb, Cr):
    Y_dct = fft.dct(fft.dct(Y, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cb_dct = fft.dct(fft.dct(Cb, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cr_dct = fft.dct(fft.dct(Cr, axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y_dct, Cb_dct, Cr_dct


#7.1.2
def idct(Y_dct, Cb_dct, Cr_dct):
    Y = fft.idct(fft.idct(Y_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cb = fft.idct(fft.idct(Cb_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cr = fft.idct(fft.idct(Cr_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y, Cb, Cr




#ortho = normalização
#axis=0 -> aplicar a dct/idct a cada linha
#axis=1 -> aplicar a dct/idct a cada coluna
#chama a função dct duas vezes para aplicar a dct a cada linha e coluna

#7.2

#7.2.1
#calcular a DCT em blocos BSxBS
def dctBlocks(Y, Cb, Cr, BS):
    Y_dct = np.zeros(Y.shape)
    Cb_dct = np.zeros(Cb.shape)
    Cr_dct = np.zeros(Cr.shape)
    for i in range(0, Y.shape[0], BS):
        for j in range(0, Y.shape[1], BS):
            Y_dct[i:i+BS, j:j+BS] = fft.dct(fft.dct(Y[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    #outro ciclo, pois após o downsampling, dimensão de Cb e Cr podem ser diferentes de Y
    for i in range(0, Cb.shape[0], BS):
        for j in range(0, Cb.shape[1], BS):
            Cb_dct[i:i+BS, j:j+BS] = fft.dct(fft.dct(Cb[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
            Cr_dct[i:i+BS, j:j+BS] = fft.dct(fft.dct(Cr[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y_dct, Cb_dct, Cr_dct

#7.2.2
#calcular a DCT inversa em blocos BSxBS
def idctBlocks(Y, Cb, Cr, BS):
    Y_dct = np.zeros(Y.shape)
    Cb_dct = np.zeros(Cb.shape)
    Cr_dct = np.zeros(Cr.shape)
    for i in range(0, Y.shape[0], BS):
        for j in range(0, Y.shape[1], BS):
            Y_dct[i:i+BS, j:j+BS] = fft.idct(fft.idct(Y[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    for i in range(0, Cb.shape[0], BS):
        for j in range(0, Cb.shape[1], BS):
            Cb_dct[i:i+BS, j:j+BS] = fft.idct(fft.idct(Cb[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
            Cr_dct[i:i+BS, j:j+BS] = fft.idct(fft.idct(Cr[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y_dct, Cb_dct, Cr_dct 


#Main
def main():
    # -----Ex.3-----
    #3.1
    #Ler imagens
    fname= "airport.bmp"
    img= plt.imread("imagens/" + fname)

    #debug print:
    print(img.shape) #tamanho da imagem
    #print(img.dtype) #tipo da imagem

    showImg(img, caption="Imagem original: " + fname)

    #3.2 colormap
    cm_red= clr.LinearSegmentedColormap.from_list("red", [(0,0,0), (1,0,0)], N= 256) #color, (range de cores rgb do vermelho)
    cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256) #256 = 2^8 (8 bits de cores)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
    cm_grey = clr.LinearSegmentedColormap.from_list("grey", [(0, 0, 0), (1, 1, 1)], N=256)

    #3.3
    #encoder
    print("Imagem [0][0]: ", img[0][0])
    R, G, B= encoder(img)
    showImg(R, cmap= cm_red, caption= "Red")
    #plt.savefig("imagens/" + fname + "_red.png") #guardar imagens
    showImg(G, cmap= cm_green, caption= "Green")
    #plt.savefig("imagens/" + fname + "_green.png") #guardar imagens
    showImg(B, cmap=cm_blue, caption="Blue")
    #plt.savefig("imagens/" + fname + "_blue.png") #guardar imagens
    #showImg(?, cmap=cm_grey, caption="Grey") fzr so dps do yCbCr

    #decoder
    #recebe rgb do encoder acima e calcula a imagem reconstruida
    imgRec= decoder(R, G, B, img.shape)
    showImg(imgRec, caption= "Imagem reconstruida: " + fname)
    #plt.savefig("imagens/" + fname + "_rec.png")

    Y , Cb, Cr= RGB_to_yCbCr(R, G, B)
    showImg(Y, cmap= cm_grey, caption= "Y")
    showImg(Cb, cmap= cm_grey, caption= "Cb")
    showImg(Cr, cmap= cm_grey, caption= "Cr")

    R, G, B= yCbCr_to_RGB(Y, Cb, Cr)
    imgRec= decoder(R, G, B, img.shape)
    showImg(imgRec, caption= "Imagem reconstruida de YCbCr: " + fname)
    print("Imagem reconstruida YCbCr [0][0]: ", imgRec[0][0])
    
    # -----Ex.6-----
    Y_d,Cb_d, Cr_d = downsampling(Y,Cb, Cr, 422)
    showImg(Cb_d, cmap= cm_grey, caption= "Cb downsampled")
    showImg(Cr_d, cmap= cm_grey, caption= "Cr downsampled")
    print("Cb downsampled size: ", Cb_d.shape)
    print("Cr downsampled size: ", Cr_d.shape)
    Y_u,Cb_u, Cr_u = upsampling(Y_d,Cb_d, Cr_d, 422)
    showImg(Cb_u, cmap= cm_grey, caption= "Cb upsampled")
    showImg(Cr_u, cmap= cm_grey, caption= "Cr upsampled")
    print("Cb upsampled size: ", Cb_u.shape)
    print("Cr upsampled size: ", Cr_u.shape)

    # -----Ex.7-----
    Y_dct, Cb_dct, Cr_dct = dct(Y_d, Cb_d, Cr_d)
    #7.1.3
    showImg(np.log(abs(Y_dct) + 0.0001), cmap= cm_grey, caption= "Y DCT")
    showImg(np.log(abs(Cb_dct) + 0.0001), cmap= cm_grey, caption= "Cb DCT")
    showImg(np.log(abs(Cr_dct) + 0.0001), cmap= cm_grey, caption= "Cr DCT")
    #7.1.4
    Y_idct, Cb_idct, Cr_idct = idct(Y_dct, Cb_dct, Cr_dct)
    print("[Y_d, Cb_d, Cr_d]", Y_d[0][0], Cb_d[0][0], Cr_d[0][0])
    print("[Y_idct, Cb_idct, Cr_idct]", Y_idct[0][0], Cb_idct[0][0], Cr_idct[0][0])

    #7.2.3
    Y_dct8, Cb_dct8, Cr_dct8 = dctBlocks(Y_d, Cb_d, Cr_d, 8)
    showImg(np.log(abs(Y_dct8) + 0.0001), cmap= cm_grey, caption= "Y DCT 8x8")
    showImg(np.log(abs(Cb_dct8) + 0.0001), cmap= cm_grey, caption= "Cb DCT 8x8")
    showImg(np.log(abs(Cr_dct8) + 0.0001), cmap= cm_grey, caption= "Cr DCT 8x8")
    Y_idct8, Cb_idct8, Cr_idct8 = idctBlocks(Y_dct8, Cb_dct8, Cr_dct8, 8)
    #print("[Y_d, Cb_d, Cr_d]", Y_d[0][0], Cb_d[0][0], Cr_d[0][0])
    print("[Y_idct8, Cb_idct8, Cr_idct8]", Y_idct8[0][0], Cb_idct8[0][0], Cr_idct8[0][0])

    #7.3
    #Fazer o mesmo que 7.2 mas com BS=64
    Y_dct64, Cb_dct64, Cr_dct64 = dctBlocks(Y_d, Cb_d, Cr_d, 64)
    showImg(np.log(abs(Y_dct64) + 0.0001), cmap= cm_grey, caption= "Y DCT 64x64")
    showImg(np.log(abs(Cb_dct64) + 0.0001), cmap= cm_grey, caption= "Cb DCT 64x64")
    showImg(np.log(abs(Cr_dct64) + 0.0001), cmap= cm_grey, caption= "Cr DCT 64x64")
    Y_idct64, Cb_idct64, Cr_idct64 = idctBlocks(Y_dct64, Cb_dct64, Cr_dct64, 64)
    #print("[Y_d, Cb_d, Cr_d]", Y_d[0][0], Cb_d[0][0], Cr_d[0][0])
    print("[Y_idct64, Cb_idct64, Cr_idct64]", Y_idct64[0][0], Cb_idct64[0][0], Cr_idct64[0][0])

    plt.show()

if __name__ == "__main__":
    main()