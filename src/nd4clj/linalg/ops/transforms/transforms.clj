(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/ops/transforms/Transforms.html"}
  nd4clj.linalg.ops.transforms.transforms
  (:refer-clojure :exclude [min max])
  (:import [org.nd4j.linalg.ops.transforms Transforms]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defn cosine-sim
  "Cosine similarity"
  [^INDArray d1 ^INDArray d2]
  (Transforms/cosineSim d1 d2))

(defn tanh [^INDArray nda]
  (Transforms/tanh nda))

(defn sigmoid [^INDArray nda]
  (Transforms/sigmoid nda))

(defn unit-vec! [^INDArray nda]
  (Transforms/unitVec nda))
