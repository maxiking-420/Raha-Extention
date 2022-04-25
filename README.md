Die Datei detection_ext.py ist eine Erweiterung für das Datenbereinigungsprogramm Raha.
https://github.com/BigDaMa/raha.git

Zur Benutzung muss die raha/detection.py Datei in Raha mit detection_ext.py ersetzt werden.
Die Erweiterung bietet die zusätzliche Label Propagation Methode "heterogenity" und
die Funktionen 'Prop7' und 'Prop8', mit gewichteter und ungewichteter Erweiterung in heterogenen Gruppen.

Als Standardkonfiguration wird zu 'Prop7' mit 'Het4' für Datensätze mit vielen heterogenen Gruppen geraten.
Auf Datensätzen mit wenig heterogenen Gruppen ist 'Prop7' ohne Erweiterung geeigneter.

Die Einstellung von SUB_CLUSTER_SIZE ist standardmäßig auf 20. Sie bestimmt, in wieviele Untergruppen eine Gruppe gespalten wird.
Ist die SUB_CLUSTER_SIZE niedrig eingestellt spart dies Rechenzeit. Eine hohe Einstellung von 100 zum Beispiel kann zu besseren Ergebnissen führen.

Zusätzlich kann mittels COMPARE_MODE der Modus für eine Metrik geändert werden. Für 'Prop8' und 'Het5' ist dies wichtig.

COMPARE_MODE = distance -> COMPARE_DISTANCE = {euclidean, hamming}

COMPARE_MODE = similarity -> COMAPRE_SIMILARITY = {matching, jaccard, dice, sneath, dot, cosine}

Falls der Datensatz sehr klein ist, kann die Anzahl an Nachbarn mit NEIGHBORS angepasst werden (Standard ist 5).

Der Ordner results enthält die Ergbnisse der Erweiterungen im Vergleich zum Standardprogramm (LABEL_PROPAGATION_METHOD = homogeneity).
Der Ordner extras enthält die Datei, mit der getestet wurde und welche Ansätze zusätzlich untersucht wurden.

Zum testen wird die Datei eva.py verwendet. Die Datei detection_ext.py muss allerdings zu testen Raha hinzugefügt werden
(Evtl. muss sie auch zu detection.py umbenannt werden).
