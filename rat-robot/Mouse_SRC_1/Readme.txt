User manual is shown in "Nermo_User_Manual.pdf" and "new_manuals.pdf".
These user manuals show how to install the robot "Nermo" and how to upload the codes.

Folder: "Sensor User  Manual" shows the detail information about the sensors that used.
Folder: "ForPCB" shows the code that need to upload for speical pcb board.

Folder: "Control" shows the code that can control the robot and should put into the raspberry pi. There are two versions, one is run in C++ and another is run in python.


The following information can be ignored.
Step 1: Create wireless hotspot
|-- Tutorials
|--|-- Win 10: https://support.microsoft.com/en-us/windows/use-your-pc-as-a-mobile-hotspot-c89b0fad-72d5-41e8-f7ea-406ad9036b85
|--|-- Ubuntu: https://help.ubuntu.com/stable/ubuntu-help/net-wireless-adhoc.html.en
|-- Wireless hotspot Setting
|--|-- SSID: LOSTLITTLEROBOT
|--|-- Passwords: mousewalk

Step 2: Connect Raspberry Pie 
|-- User: pi
|-- Passwords: mousewalk

Step 3: Control Robot
|-- Using C++ program in the Raspberry Pie
|--|-- Code of Demo: ./Source Code/Mouse Control Software
|--|-- Control commands for the Demo: Section 7 in document "Nermo_User_Manual.pdf", especially "7.4 Control"
|--|-- Robot architecture is shown in Fig. 36, and the figure is located in subsection 7.1 of "Nermo_User_Manual.pdf"


Git Links：
# NerMo v4.1
This repository contains the results of my master's thesis in informatics "A Closed Loop Walking Controller for a Biomimetic, Robtic Mouse" at TUM (March 2020 to September 2020) at and is based upon:
- [https://github.com/Luchta/nermo_robot](https://github.com/Luchta/nermo_robot)
- [https://github.com/Luchta/nermo_simulation](https://github.com/Luchta/nermo_simulation)
- [https://github.com/Luchta/nermo_code](https://github.com/Luchta/nermo_code)