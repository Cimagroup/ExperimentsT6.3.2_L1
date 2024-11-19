1) Instalación de navground en wsl. 

```bash
sudo apt install -y build-essential cmake git python3-dev python3-pip
python3 -m venv <path_to_the_venv>
. <path_to_the_venv>/bin/activate
pip install navground[all]
```

2) Para usar navground, abrir esta carpeta(la carpeta base donde hayais hecho todos los pasos anteriores) en el cmd, poner wsl. 

	1. Activar el entorno virtual para ello: source entorno/bin/activate (como se llame tu entorno)

	2. Instalar  librerías necesarias para poder ejecutar los notebooks: pip install jupyter notebook matplotlib scipy multiprocess gudhi plotly scikit-learn pandas (no es necesario instalar cairosvg websockets y moviepy, ya que se instalan con navground)

	3. Abrir jupyter notebook, puedes seguir de ejemplo el archivo CorridorScenario.ipynb o el archivo CrossScenario. El tour.ipynb es 	otro archivo tutorial, lo puedes descargar la página de navground. 

	**Si da problema cairo durante la ejecución del notebook: sudo apt install cairo2**

En caso de problemas con wsl, reinstalar con la siguiente distribución y volver a instalar navground con los pasos indicados anteriormente:

1) Eliminar distribucion ubuntu completa wsl --unregister distribution

2) Instalar distribucion wsl --install -d Ubuntu-22.04

3) sudo apt-get update

4) wsl --set-default-version 2

