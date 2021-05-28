import setuptools
 
package_name = 'precept'

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req:
    requirements = req.read().splitlines()
 
setuptools.setup( name                          = package_name
                , version                       = '0.2.0'
                , author                        = 'Yannick Uhlmann'
                , author_email                  = 'yannick.uhlmann@reutlingen-university.de'
                , description                   = 'Deep Learning based Primitive Device Approximation'
                , long_description              = long_description
                , long_description_content_type = 'text/markdown'
                , url                           = 'https://github.com/electronics-and-drives/precept'
                , packages                      = setuptools.find_packages()
                , classifiers                   = [ 'Development Status :: 2 :: Pre-Alpha'
                                                  , 'Programming Language :: Python :: 3'
                                                  , 'Operating System :: POSIX :: Linux' ]
                , python_requires               = '>=3.8'
                , install_requires              = requirements
                , entry_points                  = { 'console_scripts': [ 'pct = precept.__main__:pct' 
                                                                       , 'prc = precept.__main__:prc']}
                , package_data                  = { '': ['*.hy', '__pycache__/*']}
                , data_files                    = [ ('share/man/man1', ['doc/pct.1'])
                                                  , ('share/man/man8', ['doc/precept.8'])]
                , )
