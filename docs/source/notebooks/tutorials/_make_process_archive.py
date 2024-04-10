#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 10/04/2024, 13:11. Copyright (c) The Contributors

# This script will create and process (XMM process anyway) an archive of observations of Abell 3667, so I have
#  something to load in for the archives tutorial

from daxa.archive import Archive
from daxa.mission import XMMPointed, Chandra, eRASS1DE, ROSATPointed
from daxa.process.simple import full_process_xmm

xm = XMMPointed()
ch = Chandra()
er = eRASS1DE()
rp = ROSATPointed()

xm.filter_on_name("A3667")
ch.filter_on_name("A3667")
er.filter_on_name("A3667")
rp.filter_on_name("A3667")

xm.download()
ch.download(download_products=True)
er.download()
rp.download(download_products=True)

arch = Archive("A3667_made_earlier", [xm, ch, er, rp])

full_process_xmm(arch)
