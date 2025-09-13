import os, time, jwt, sys


secret = os.getenv("JWT_SECRET", "dev-secret")
sub = sys.argv[1] if len(sys.argv) > 1 else "user"
exp = int(time.time()) + 3600
token = jwt.encode({"sub": sub, "exp": exp}, secret, algorithm="HS256")
print(token)
