(import [numpy :as np])

(defn scl [x &optional [a 0.0] [b 1.0]]
f"Feature normalization (scaling)
Takes a 1D vector and interval [a,b] and scales it according to:

             ⎛  x-min(x)   ⎞  
  x' = (b-a)∙⎜―――――――――――――⎟+a
             ⎝max(x)-min(x)⎠  

Returns the normalzied vector x'∈ [a,b]
"
  (+ (* (- b a) (/ (- x (np.min x)) (- (np.max x) (np.min x)))) a))

(defn bct [y &optional [λ 0.2]]
f"Box-Cox Transformation
Takes a scalar or 1D np.array and transforms it
according to:

        λ  
       y -1
  y' = ――――   if λ ≠ 0
        λ  

  y' = ln(y)  if λ = 0

Returns the transformed scalar or vector.
"
  (if (= λ 0)
    (np.log y)
    (/ (- (np.power y λ) 1) λ)))

(defn cbt [y′ &optional [λ 0.2]]
f"Inverse Box-Cox Transformation (Cox-Box)
Takes a scalar or 1D np.array and transforms it
according to:

       ⎛ln(y'∙λ+1)⎞
       ⎜――――――――――⎟
       ⎝    λ     ⎠
  y = e               if λ ≠ 0

       y'
  y = e               if λ = 0

Returns the inverse transformed scalar or vector.
"
  (if (= λ 0)
    (np.exp y′)
    (np.exp (/ (np.log (+ (* y′ λ) 1)) λ))))
