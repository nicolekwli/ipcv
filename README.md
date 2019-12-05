# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
Google how to install open cv 
I found one that teaches how to install open cv 4, and also open cv 2

Install XCode
Install Brew with /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install opencv
brew install pkg-config

* Configuration
See if you have linker flags for opencv with 
pkg-config --cflags --libs opencv

If you dont have a bunch of file paths, you find our where opencv.pc is

and use this: pkg-config --cflags --libs path/to/opencv.pc
 in this: g++ $(pkg-config --cflags --libs opencv) -std=c++11  yourFile.cpp -o yourFileProgram

* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
