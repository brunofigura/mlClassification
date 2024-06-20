# Thema 2 - Klassifizierung

## Setup
#### 1. Schritt: Das GitHub repository klonen
Im Terminal soll zunächst ins Ziel-Directory navigiert werden.
Anschließend kann über den Befehl
`git clone https://github.com/brunofigura/mlClassification.git`
das Projekt heruntergeladen werden.

#### 2. Schritt - Virtual Enviroment erstellen
Da das venv sehr groß ist, ist es nicht im repository hinterlegt und muss manuel initiert werden.

Auf Windows:
`c:\>python -m venv c:\path\to\myenv`

Auf MacOs oder Linux:
`python -m venv /path/to/new/virtual/environment`

Anschließend sollte das Virtual Enviroment gestartet werden.
Auf Windows in der **PowerShell**:
`PS C:\> <venv>\Scripts\Activate.ps1`

Auf MacOs oder Linux
`$ source <venv>/bin/activate`

#### 3. Schritt - Packages installieren
Auch die benutzen Libraries wurden nicht auf GitHub gespeichert. Sie können allerdinges ganz einfach lokal installiert werden. 

Alle benutzten Packages wurden in der _requirements.txt_ Datei gespeichert.
Der Befehl `pip install -r requirements.txt` installiert alle benutzen Packages des Projektes in dem Virtual Enviroment.

> Es muss vorher sicher gestellt werden, dass das man sich im Projekt-Directory befindet

## Netze trainieren und evaluieren

Jeder Klassifizierer hat sein eigenes Python-Programm. Führt man diese in einer IDE aus, wird das entsprechende Netz trainiert. 
Jedes Programm folgt dem gleichen Aufbau.
> 1. Netz modellieren
> 2. Klassifizierer modellieren, der ein Netz trainieren, validieren, testen und Auswerten kann
>  3. Eine Main-Methode die beim Programm-Start ausgeführt wird.

In der Main-Methode gibt man alle relevanten Hyperparameter an. Es besteht auch die Möglichkeit nach dem Training einen Netzes, die trainierten Gewichte in einer Datei zu speichern und diese zu einem späteren Zeitpunkt wieder zu laden. 

Achtung! Aus Speichergründen sind die trainierten Gewichte nicht da, sondern müssten erstmal generiert werden. 

