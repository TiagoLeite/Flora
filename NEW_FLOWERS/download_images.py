import os


file = open('flowers_latin.txt')

cont = 0

lines = file.readlines()

print('Size: ' + str(len(lines)))

for line in lines:
    specie = line.replace("\n", '')
    print("\n===== " + specie + " (" + str(cont) + ")" + " ========")
    os.system("googleimagesdownload --keywords " + "\"" + specie + " flower\"" + " --limit 100")
    # os.system("googleimagesdownload --keywords " + "\"" + specie + " flower closer\" --limit 100 --size large")
    # os.system("googleimagesdownload --keywords " + "\"" + specie + " flower closer\" --limit 100 --size medium")
    cont += 1

# última: Unxia suffruticosa

"""Antes:
Agapanthus africanus
Lobularia maritima
Alstroemeria x hibrida
Hippeastrum hybridum
Viola x wittrockiana
Anthurium andraeanum
Symphyotrichum tradescantii
Rhododendron simsii
Begonia semperflorens
Impatiens hawkeri
Antirrhinum majus
Bellis perennis
Unxia suffruticosa
"""
