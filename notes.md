# Table of contents
0. [Edge Detector](#Edge Detector)
1. [Hough Transform](#Hough Transform)
2. [Harris Corner Detector](#Harris Corner Detector)
3. [Image Rotation](#Image Rotation)

## Edge Detector
---
> [!question] What is an Edge?
> A rapid change in image intensity within small region

We want an **Edge Operator** that produces:
- Edge *Position*
- Edge *Magnitude* (strength)
- Edge *Orientation* (direction)

Performance Requirements:
- High *Detection Rate*
- Good *Localization*
- Low *Noise Sensitivity*

### 

---
Edge Detection using Gradients
Edge Detection using Laplacian
Canny Edge Detector
Corner Detection
## Hough Transform



## Harris Corner Detector

## Image Rotation


3 Το πρόβλημα Σε αυτή την ενότητα θα χρησιμοποιήσετε τις τρομερές γνώσεις που έχετε
αποκτήσει μέχρι τώρα στην ψηφιακή επεξεργασία εικόνας καθώς και τις συναρτήσεις που
έχετε φτιάξει σε αυτή την εργασία για να λύσετε το παρακάτω πρόβλημα. Θεωρήστε πως
η εικόνα εισόδου που σας δόθηκε δημιουργήθηκε από μια άτακτη τοποθέτηση εγγράφων
πάνω στην επιφάνεια ενός scanner. Παρά την αδιανόητη τεμπελιά μας θα θέλαμε να
έχουμε μια ψηφιοποιήμενη έκδοση κάθε εγγράφου που τοποθετήσαμε στο scanner. Για να
γίνει αυτό, ο αλγόριθμος σας θα πρέπει να εντοπίζει κάθε ένα από τα έγγραφα τις εικόνας
(ή όσα περισσότερα μπορείτε), να τα απομονώνει, στην συνέχεια να τα περιστρέφει
κατάλληλα και τέλος να τα αποθηκεύει σαν ανεξάρτητες εικόνες στο ίδιο format με την
εικόνα εισόδου και πρόσθετη κατάληξη _1, _2 κ.λπ. Φτιάξτε το Python script με όνομα 
my_lazy_scanner.py το οποίο θα φορτώνει την εικόνα εισόδου με στόχο την ανεξάρτητη 
ψηφιοποίηση των εγγράφων της. Το script θα πρέπει να ζεί στο ίδιο directory με τις 
συναρτήσεις σας. Το script πρέπει να μας πληροφορεί πόσες εικόνες εντόπισε και πόσες
τελικά πέτυχε να γράψει στο δίσκο σαν αυτόνομα αρχεία. Να δείξετε την λειτουργία της 
μεθόδού σας και για τις πέντε εικόνες που σας δίνονται μαζί με την εκφώνηση. Τέλος, 
είναι απαραίτητο το script σας να λειτουργεί χωρίς a-priori γνώση της εικόνας εισόδου και
να λειτουργεί με τις ίδιες παραμέτρους για όλες τις εικόνες εισόδου (one-script-fits-all 
λογική).


Στοχοι
1. εντοπισμος εγγραφων (how many was it able to find and then store)
2. απομονωση
3. περιστροφη
4. αποθηκευση

Harris corner detector
input -> gray scale image
output -> 