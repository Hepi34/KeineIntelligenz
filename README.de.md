# KeineIntelligenz

## Einleitung
Dies ist KeineIntelligenz, ein Programm zum Trainieren eines CNN auf dem MNIST-Datensatz. Es ist in Python geschrieben. Es werden sowohl CPU-Training als auch GPU-Training über OpenCL (PyOpenCL) unterstützt. Das Programm wurde unter Windows und macOS getestet und sollte auch unter Linux funktionieren.

## Nutzung
Um das Programm zu verwenden, klone zuerst dieses Repository und starte dann `start.py` aus dem Projektstamm:

- Windows: `python start.py`
- macOS/Linux: `python3 start.py`

Falls nötig, wird automatisch eine virtuelle Umgebung erstellt und die erforderlichen Abhängigkeiten installiert. Danach öffnet sich die Haupt-GUI.

## GUI

Die GUI bietet eine benutzerfreundliche Oberfläche zum Trainieren und Testen des neuronalen Netzes. Du kannst Modelle erstellen, Trainingseinstellungen anpassen und den Trainingsfortschritt in Echtzeit verfolgen.

## Code-Dokumentation

### Dateiübersicht

- **`start.py`** — Startet das Programm
- **`pyfiles/gui.py`** — Die grafische Benutzeroberfläche
- **`pyfiles/model.py`** — Das neuronale Netzmodell
- **`pyfiles/trainer.py`** — Trainingslogik
- **`pyfiles/dataset.py`** — Lädt MNIST-Daten
- **`pyfiles/layers.py`** — Bausteine (Conv, Dense usw.)
- **`pyfiles/loss.py`** — Verlustfunktion für das Training
- **`pyfiles/optimizers.py`** — Optimierer (SGD, Adam)
- **`pyfiles/utils.py`** — Hilfsfunktionen
- **`pyfiles/gpu_pipeline.py`** — GPU-Beschleunigung
- **`pyfiles/opencl_backend.py`** — OpenCL-Setup
- **`pyfiles/opencl_layers.py`** — GPU-Layer-Implementierungen

### Dateibeschreibungen

#### start.py

Der Haupteinstiegspunkt, der vor dem Start alles vorbereitet. Er:
- Prüft, ob eine virtuelle Umgebung existiert (`.venv`-Ordner)
- Erstellt sie, falls sie nicht existiert (mit `python3 -m venv` auf macOS/Linux oder `python -m venv` unter Windows)
- Liest `requirements.txt` und prüft, welche Pakete bereits installiert sind
- Installiert fehlende Pakete mit pip
- Startet schließlich `pyfiles/gui.py` mit der Python-Version aus der virtuellen Umgebung

Starte diese Datei, wann immer du die GUI öffnen möchtest. Fehlende Abhängigkeiten werden automatisch ergänzt.

#### pyfiles/gui.py

Die grafische Oberfläche, gebaut mit PyQt6. Sie bietet:
- **Presets** - Vorkonfigurierte Modell-Setups (v1/Mini, v1/Normal, v1/Pro, v2/Mini, usw.), die du schnell auswählen kannst. Jedes Preset hat unterschiedliche Trainingseinstellungen, Netzwerkgrößen und Datenlimits.
- **Datenladen** - Findet und lädt MNIST-Bilder und -Labels automatisch aus dem Dataset-Ordner
- **Fortschrittsanzeige** - Zeigt den Trainingsfortschritt mit Graphen, Verlustkurven und Genauigkeit über die Zeit
- **GPU-Erkennung** - Erkennt automatisch, ob eine GPU verfügbar ist, und kann sie für schnelleres Training nutzen
- **Modell speichern/laden** - Speichert trainierte Modelle als .npz-Dateien und lädt sie später wieder
- Nutzt Threads, damit die UI während des Trainings reaktionsfähig bleibt

#### pyfiles/model.py

Der Container, der alle Layer zu einem vollständigen Netzwerk zusammenfasst. Die Klasse `CNNModel` übernimmt:
- **Forward Pass** - Nimmt Eingabedaten und leitet sie durch alle Layer (Conv2D → ReLU → MaxPool → ... → Dense)
- **Backward Pass** - Geht die Layer rückwärts durch, um Gradienten zu berechnen (für das Aktualisieren der Gewichte)
- **Parameterverwaltung** - Sammelt alle Gewichte und Biases aus allen Layern, damit der Optimierer sie aktualisieren kann
- **Gradientenverwaltung** - Sammelt alle Gradienten aus allen Layern, damit der Optimierer weiß, wie stark die Gewichte angepasst werden müssen

Beispiel: Wenn du [Conv2D → ReLU → Dense] hast, verkettet das Modell diese Layer und kümmert sich um den Datenfluss.

#### pyfiles/trainer.py

Steuert den gesamten Trainingsprozess. Es enthält:
- **TrainConfig** - Speichert Einstellungen wie Anzahl der Epochen, Batchgröße, ob die Daten gemischt werden, und die Threadanzahl
- **Trainer-Klasse** - Führt das Training in folgenden Schritten aus:
  1. Schleife über jede Epoche
  2. Für jedes Daten-Batch: Forward Pass, Loss berechnen, Backward Pass, Gewichte aktualisieren
  3. Nach jeder Epoche: Auf Testdaten evaluieren und Genauigkeit speichern
  4. Historie von Loss und Genauigkeit zurückgeben
- **Batch Iterator** - Teilt die Daten in kleine Batches für das Training
- **Genauigkeitsberechnung** - Prüft, wie viele Vorhersagen korrekt sind
- **Thread-Verwaltung** - Kann mehrere CPU-Kerne für schnellere NumPy-Operationen nutzen

#### pyfiles/dataset.py

Lädt den MNIST-Datensatz mit handgeschriebenen Ziffern. Es:
- **Liest IDX-Format** - MNIST-Dateien sind ein spezielles Binärformat (.ubyte). Der Code liest:
  - Magic Numbers zur Dateitypprüfung
  - Bildanzahl, Dimensionen (28×28 für MNIST)
  - Roh-Pixelbytes (ein Byte pro Pixel = 0-255)
  - Label-Bytes (0-9 für Ziffern)
- **Normalisiert Daten** - Konvertiert Pixelwerte von 0-255 auf 0-1 durch Division durch 255 (hilft beim Training)
- **Formt Bilder um** - Konvertiert flache Byte-Arrays in 4D-Arrays (N, 1, 28, 28), wobei N = Bildanzahl
- **Teilt Daten** - Gibt Trainings- und Testbilder sowie Labels als separate Arrays zurück
- **MNISTDataset-Klasse** - Einfacher Container, der alle vier Arrays bündelt

#### pyfiles/layers.py

Definiert die Bausteine des neuronalen Netzes. Jeder Layer hat `forward()` (Ausgabe berechnen) und `backward()` (Gradienten berechnen):

- **Conv2D** - Convolution-Layer, der Muster in Bildern findet:
  - Verwendet 3×3-Filter, die über das Bild gleiten
  - Hat Stride- und Padding-Optionen
  - Speichert Gewichte und Biases, die beim Training aktualisiert werden
  - Nutzt "He Initialization" für sinnvoll skalierte Startgewichte

- **Dense** - Voll verbundenes Layer, das jeden Input mit jedem Output verbindet:
  - Wie eine große Matrixmultiplikation mit Bias
  - Wird meist am Ende vor der finalen Vorhersage genutzt
  - Nimmt 2D-Input (Batch, Features) und liefert (Batch, Outputgröße)

- **ReLU** - Aktivierungsfunktion für Nichtlinearität:
  - Einfach: Ausgabe = max(0, Eingabe)
  - Setzt negative Werte auf 0, lässt positive Werte unverändert
  - Macht das Netzwerk fähig, nichtlineare Muster zu lernen

- **MaxPool2D** - Reduziert die Größe und behält wichtige Informationen:
  - Schiebt ein Fenster (z. B. 2×2) über das Bild
  - Behält nur den Maximalwert in jedem Fenster
  - Reduziert Rechenaufwand und hilft gegen Overfitting

- **Flatten** - Konvertiert 2D/3D-Daten zu 1D:
  - Nimmt (N, C, H, W) und formt zu (N, C*H*W) um
  - Brücke zwischen Convolution-Layern und Fully-Connected-Layern

- **Dropout** - Entfernt während des Trainings zufällig Neuronen:
  - Hilft gegen Overfitting durch "Ausdünnen" des Netzes
  - Nur während des Trainings aktiv, beim Testen deaktiviert
  - Skaliert die verbleibenden Aktivierungen zur Kompensation

#### pyfiles/loss.py

Misst, wie falsch die Vorhersagen des Netzes sind. Es enthält:

- **CrossEntropy** - Die Loss-Funktion für Ziffernklassifikation:
  - Nimmt Rohvorhersagen (Logits) und echte Labels (0-9)
  - Berechnet Softmax (wandelt in Wahrscheinlichkeiten mit Summe 1 um)
  - Misst die Abweichung der vorhergesagten Wahrscheinlichkeiten von der Wahrheit (hoher Loss = schlechte Vorhersagen)
  - **Numerische Stabilität** - Verwendet einen Trick (Max abziehen vor exp), um Überläufe zu vermeiden
  - **Backward** - Berechnet Gradienten für die Gewichts-Updates
  - Niedrigerer Loss = bessere Vorhersagen = happy training!

#### pyfiles/optimizers.py

Enthält Algorithmen, die das Netzwerk beim Training verbessern, indem sie Gewichte aktualisieren:

- **SGD (Stochastic Gradient Descent)** - Einfacher Optimierer:
  - Aktualisiert Gewichte durch: `weight -= learning_rate * gradient`
  - Alle Parameter werden gleich behandelt
  - Kann Weight Decay (L2-Regularisierung) hinzufügen, um Overfitting zu verhindern
  - Schnell, braucht aber manchmal Feintuning

- **Adam** - Fortschrittlicher Optimierer:
  - Passt Lernraten pro Parameter individuell an
  - Behält Momentum (merkt sich frühere Updates)
  - Braucht meist weniger Feintuning als SGD
  - Konvergiert in der Praxis oft besser
  - Hat Parameter: beta1 (Momentum für Gradienten), beta2 (Momentum für quadrierte Gradienten), epsilon (numerische Stabilität)

Beide können an den Trainer übergeben werden, zusammen mit einer Lernrate.

#### pyfiles/utils.py

Hilfsfunktionen und Utilities, die im Training genutzt werden:

- **accuracy_score()** - Berechnet, wie viele Vorhersagen korrekt sind:
  - Vergleicht vorhergesagte Klasse mit dem wahren Label
  - Gibt den Prozentanteil korrekter Vorhersagen zurück

- **batch_iterator()** - Erstellt Mini-Batches fürs Training:
  - Teilt Trainingsdaten in kleinere Batches auf
  - Kann Daten vor dem Batching zufällig mischen
  - Liefert ein Batch nach dem anderen, damit nicht alles auf einmal geladen werden muss

- **Timer** - Einfache Utility zum Messen von Laufzeiten:
  - `start()` - Startet die Zeitmessung
  - `stop()` - Beendet sie und liefert verstrichene Sekunden
  - Nützlich zum Messen der Epochenzeit oder Gesamttrainingsdauer

#### pyfiles/gpu_pipeline.py

Eine vollständige GPU-Trainingspipeline mit OpenCL. Sie enthält:

- **OpenCL-Kernel** - Low-Level-GPU-Code in C-ähnlicher Sprache, der auf der GPU läuft:
  - `conv2d_forward` - Convolution-Forward-Pass für die GPU optimiert
  - `conv2d_backward` - Convolution-Backward-Pass für Gradienten
  - `dense_forward` - Fully-Connected-Forward
  - `dense_backward` - Fully-Connected-Backward
  - `copy_batch` - Schnelles Kopieren von Batches im Speicher
  - Weitere Kernel für Aktivierungsfunktionen, Pooling, Dropout

- **GPUTrainConfig** - Einstellungen für GPU-Training (Epochen, Batchgröße, Layergrößen, Optimizerwahl, usw.)

- **GPUTrainingPipeline** - Der GPU-Trainer, der:
  - OpenCL-Kernel einmalig beim Start kompiliert
  - Daten zwischen CPU- und GPU-Speicher überträgt
  - GPU-Kernel für Forward/Backward-Passes startet
  - Parameter-Updates auf der GPU durchführt
  - Für große Modelle deutlich schneller ist (bei kleinen Modellen ggf. Overhead)

Diese Datei ist eine reine GPU-Implementierung, während die CPU-Version das normale `trainer.py` nutzt.

#### pyfiles/opencl_backend.py

Richtet OpenCL für die GPU-Beschleunigung ein. Es:

- **Importiert PyOpenCL** - Die Bibliothek, die mit GPUs kommuniziert
- **GPUDeviceInfo** - Dataclass mit GPU-Infos (Plattformname, Gerätename, Hersteller, Treiberversion)
- **OpenCLManager** - Hauptklasse, die:
  - Verfügbare GPU-Geräte erkennt (NVIDIA, AMD, Intel, Apple Silicon, usw.)
  - Einen OpenCL-Kontext erstellt (Verbindung zur GPU)
  - Eine Command Queue erstellt (schickt Befehle an die GPU)
  - Hilfsmethoden bereitstellt, um Tensoren in/aus GPU-Speicher zu übertragen
  - Buffer für Daten auf der GPU verwaltet

Beim Programmstart wird versucht, eine GPU zu erkennen. Wenn eine gefunden wird, kann die GUI GPU-Training anbieten.

#### pyfiles/opencl_layers.py

GPU-Versionen der neuronalen Layer mit OpenCL. Statt NumPy auf der CPU laufen diese Layer:
- Vollstaendig auf der GPU fuer Geschwindigkeit
- Enthalten OpenCL-Kernel (GPU-Code) fuer jede Operation
- Werden von der GPUTrainingPipeline statt der normalen CPU-Layer genutzt
- Enthalten GPU-Versionen von Conv2D, Dense, ReLU, MaxPool2D und weiteren Operationen
- Verwalten Speichertransfer zwischen CPU und GPU
