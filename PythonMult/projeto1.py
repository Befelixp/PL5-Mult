import matplotlib.pyplot as plt #3.1
import matplotlib.colors as clr #3.2
import numpy as np #decoder

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
    cm_green = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (0, 1, 0)], N=256) #256 = 2^8 (8 bits de cores)
    cm_blue = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (0, 0, 1)], N=256)
    cm_grey = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 1, 1)], N=256)

    #3.3
    #encoder
    R, G, B= encoder(img)
    showImg(R, cmap= cm_red, caption= "Red")
    plt.savefig("imagens/" + fname + "_red.png") #guardar imagens
    showImg(G, cmap= cm_green, caption= "Green")
    plt.savefig("imagens/" + fname + "_green.png") #guardar imagens
    showImg(B, cmap=cm_blue, caption="Blue")
    plt.savefig("imagens/" + fname + "_blue.png") #guardar imagens
    #showImg(?, cmap=cm_grey, caption="Grey") fzr so dps do yCbCr

    #decoder
    #recebe rgb do encoder acima e calcula a imagem reconstruida
    imgRec= decoder(R, G, B, img.shape)
    showImg(imgRec, caption= "Imagem reconstruida: " + fname)
    plt.savefig("imagens/" + fname + "_rec.png")

    plt.show()

if __name__ == "__main__":
    main()