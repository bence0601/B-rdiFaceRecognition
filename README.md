Face Recognition project for Bárdi autó.

So far:
        -Can recognize face using webcam, if it has a picture of the specific person
        -Can detect if someone is not on the pic trynna access, and it will say 'MATCH' or 'NO MATCH' based on the result

Thing that need to be implemented:
                                  - Now it's working on local pc, but using Flask for python, you can deploy tensorflow, so you can access it via Google etc.
                                  - How to detect the difference between a photo and real person: While sampling the person, the app should check the vectors of the pixels of the  face. If all pixels' vector moving at the same time at the same directon, then it's a photo.
                                  - Use different directories, for positive, negative, and anchor picturer for the learning process, based on: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
