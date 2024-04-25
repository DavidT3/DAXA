# Publishing a new version of DAXA

This very short guide is for my own benefit, so I'll remember how I set all of this up the next time I need to do this. The process is largely automated now, but there are a couple of steps to it that have to be done in the right way.

I TRIED TO MAKE THIS IDENTICAL TO XGA, and in some ways it is hopefully better, and in one significant way it is much worse. The improvement is that hopefully versions of the built DAXA wheel sent to PyPI will be correct, as it is set directly from the tag, and we're using PyPI's 'trusted publisher' functionality. However the downside is that 'daxa' is taken on testpypi, and I can't make it publish to a differently named project on testpypi, so it just doesn't work for now.

## The steps
1) Check that all dependencies in setup.py and requirements.txt are up to date and correct.
2) Check that no new files or directories need to be included in the MANAFEST.in
3) Checkout the master branch on local machine, then create an annotated tag (git tag -a v0.1.1 -m "MESSAGE HERE"). Then push that tag to the remote (either git push origin v0.1.1 or using Pycharm).
4) This will trigger the build and publishing to test PyPI - THIS WILL FAIL BECAUSE OF THE DOWNSIDE I MENTIONED ABOVE
5) STOP AND BE SAD BECAUSE THIS WON'T WORK - Now check that the install from ```pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple daxa-xray``` 
6) Now do a release on the GitHub website - this should trigger the build and publishing to the real PyPI index.

## Notes to self
* Seems like sometimes I need to make sure to push tags from all branches from PyCharm
* It will show up as a release but it doesn't seem to trigger the PyPI release action
* To delete a remote tag (as I have had to a LOT while figuring this all out), use git push --delete origin {tag_name}
