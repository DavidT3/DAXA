#
#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by David J Turner (djturner@umbc.edu) 10/9/26, 9:52 AM. Copyright (c) The Contributors.
#

coverage run -m unittest discover
coverage report
coverage xml
#coverage html

genbadge coverage -i coverage.xml


