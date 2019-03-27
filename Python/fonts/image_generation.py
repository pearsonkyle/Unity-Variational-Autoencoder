import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    plt.style.use('dark_background')
    prop = fm.FontProperties(fname='Pharoah Glyph Medium.ttf')
    #prop = fm.FontProperties(fname='MERO_HIE.TTF')
    #prop = fm.FontProperties(fname='Prehistoric Paintings.ttf')
    #prop = fm.FontProperties(fname='mayanglyphsfill-Regular.ttf')

    lower = 'abcdefghijklmnopqrstuvxyz'
    upper = 'abcdefghijklmnopqrstuvxyz'.upper()
    numbers = '1234567890-='
    special = '!@#$%^&*()_+'

    allchars = lower+upper+numbers+special
    my_dpi = 125

    i = 0
    for x in range(10):
        for y in range(np.round(len(allchars)/10).astype(int)):
            fig, ax = plt.subplots(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        
            ax.text(0,0, allchars[i], fontproperties=prop, size=400, color='white')
            ax.plot([0,0.1],[0,0.1],color='black')
            ax.set_xlim([0,0.1])
            ax.set_ylim([0,0.1])
            plt.axis('off')
            plt.savefig("Hieroglyphs/{}_large.png".format(allchars[i]) )
            plt.close()

            i+=1
            if i==len(allchars):
                break
        
        if i==len(allchars):
            break