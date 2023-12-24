# Using the VirtualBox computational environment

## First use

Vagrant and VirtualBox exist for most operating systems, including MS Windows.

In your BIOS, you may need to turn off secure boot and allow Virtualization Technology (VT), which may be disabled by default.

Assuming a recent Ubuntu system:

1. `sudo apt install virtualbox vagrant virtualbox-ext-pack virtualbox-guest-utils git`
2. `git clone git@github.com:joachimvandekerckhove/ezbhddm.git`
3. `cd ezbhddm/vm && vagrant up` (and note the jupyter token)
4. Point browser to `https://localhost:21088/`


