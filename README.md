PostScan Helper
================

PostScan Helper is a simple utility to help you deal with scanned photograph.
It helps most when you put multiple photos to scanner, as it can extract them
to separate images, crop them, fix their rotation, optimize white balance and
all of that in both interactive and non-interactive mode.

Installation
------------

You will need installed:
* Python 3.x
* OpenCV
* ImageMagick (optional)
* Then you can `pip install -r requirements.txt`

Wand library is optional, it complements image loading,
and you will need it only if ordinary OpenCV cannot
load your images (it can happen with some older TIFFs...) 

Usage
-----

* Have directory `inputdir` with scanned images
* Have empty directory `outputdir` where images will be saved
* Run `python3 ./postscanhelper.py inputdir -o outputdir`
* You will be presented with images from `inputdir`, one by one.
  * Press any key to proceed
  * Press `i` to ignore this image.
* If you have option `--threshold-tweak`, next image will show threshold level.
  * Control with `+` and `-` keys until you are satisfied.
Press any other key to continue.
* Now you will get cropped image that software detected. On this dialog, you can:
  * `s` - saves image to output directory
  * `r` - rotate image
  * `a` - do auto enhance (fix white balance)
  * `b` - increase borders by additional 1px (crops image)
  * `u` - resets (undo) image to original one
  * `<Esc>`/`i` - skips (ignores) current image
  * `q` - quits application

Closing popup dialog at any time will completely close app.

More hints:
* If you want to run app without showing any GUI dialogs, use `--non-interactive`
* Run `python3 ./postscanhelper.py --help` to see all options

Example
-------

We are starting with bunch of old photos stacked together to scanner.
Scanner is set to A3, high quality color output and to TIFF format.

Image that we will feed to program looks like this:

![Scan](docs/1.png?raw=true "Scan")

You will notice I accidentally turned it upside down, photos are bit rotated and
also photos started getting nice sepia effect (no photoshop here!). But, as long
as you have your background mostly white, app will figure out the rest.

If you added `--threshold-tweak`, next dialog you will get is this:

![Threshold](docs/2.png?raw=true "Threshold")

With `+` and `-`, you can "fix" default threshold values,
but what you see here is what it is supposed to look like:)
You will have to fiddle with this, only if photos are not getting recognized.

After you are done with this, app will start showing you extracted images one by one.
There you can tweak it more. For example, one of the pictures will be automatically
rotated based on image context (yes, it is that smart!), cropped properly,
border will be cut to remove scanning artifacts and white balance will be fixed.
End result will be shown, and it should look something like this:

![Result](docs/3.png?raw=true "Result")