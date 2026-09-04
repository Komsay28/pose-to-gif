Pose-to-GIF Matcher

A real-time computer vision app: strike a pose in front of your webcam, and it finds the closest matching GIF(s) from a pre-indexed library — using pose landmark detection and cosine similarity, not raw image comparison.


Tech stack
Pose detection: MediaPipe Pose Landmarker (Tasks API)
Computer vision / video I/O: OpenCV
GIF processing: Pillow
Matching: Cosine similarity (pure Python)
Language: Python 3

Scope & limitations
Matching relies on detecting a human body pose, so it only works reliably on GIFs featuring a clearly visible, mostly full-body human subject. Cartoons, close-up/face-only reaction GIFs, animals, and object-only GIFs generally won't produce usable pose data and are skipped during database processing.
Match search is throttled (not run on every single webcam frame) to keep the live feed responsive during brute-force comparison against the database.
Currently a local desktop demo (OpenCV window); a web-based version (FastAPI backend + browser-based camera/canvas frontend) is planned.
Roadmap
 Web interface (FastAPI backend, HTML canvas frontend)
 Expand GIF library
 Explore approximate/indexed search for larger libraries (currently brute-force)
 Investigate non-pose-based matching for non-human GIFs
Author

Built by Komsay28 as a learning project in computer vision and deep learning.
