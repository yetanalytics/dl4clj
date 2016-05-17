(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/api/ndarray/INDArray.html"}
  nd4clj.linalg.api.ndarray.indarray
  (:refer-clojure :exclude [min max])
  (:import [org.nd4j.linalg.api.ndarray INDArray]))

;;; Matrix manipulation
;;; -------------------

(defn put-scalar 
  "Insert the item at the specified indices"
  [^INDArray a indices value]  
  (.putScalar a (int-array indices) value))

(defn put-row
  "Insert a row in to this array Will throw an exception if this ndarray is not a matrix"
  [^INDArray a row to-put]
  (.putRow a (int row) to-put))

;;; matrix properties
;;; -----------------

(defn is-row-vector [^INDArray a]
  (.isRowVector a))

(defn is-column-vector [^INDArray a]
  (.isColumnVector a))

(defn shape
  "Returns the shape of an ndarray"
   [^INDArray a]
  (into [] (.shape a)))

(defn norm1 ^Number [^INDArray v]
  (.norm1Number v))

(defn norm2 ^Number [^INDArray v]
  (.norm2Number v))

(defn distance1 [^INDArray v1 ^INDArray v2]
  (.distance1 v1 v2))

(defn distance2 [^INDArray v1 ^INDArray v2]
  (.distance2 v1 v2))

;;; matrix values
;;; -------------

(defn get-scalar 
  "Get the vector along a particular dimension"
  [^INDArray a indices]
  (.getScalar a (int-array indices)))

(defn tensor-along-dimension [^INDArray a index dimensions]
  (.tensorAlongDimension a (int index) (int-array dimensions)))

(defn get-double [^INDArray a indices]
  (.getDouble a (int-array indices)))

(defn data
  "Returns a linear double array representation of this ndarray"
  [^INDArray a]
  (let [da (double-array (.length a))]
    (dotimes [i (.length a)]
      (aset da i (.getDouble a (int i))))
    (into [] da)))

(defn slice
  "Returns the specified slice of this ndarray"
  ([^INDArray a i]
   (.slice a (int i)))
  ([^INDArray a i dimension]
   (.slice a (int i) (int dimension))))

;;; Matrix reductions
;;; -----------------

(defn max
  "Returns the overall max of this ndarray"
  [^INDArray this dimension]
  (.max this (int-array [dimension])))

(defn mean
  "Returns the overall mean of this ndarray"
  [^INDArray this dimension]
  (.mean this (int-array [dimension])))

(defn min
  "Returns the overall mean of this ndarray"
  [^INDArray this dimension]
  (.min this (int-array [dimension])))


;;; Matrix duplication
;;; ------------------

(defn dup 
  "Return a copy of an INDarray"
  [^INDArray a]
  (.dup a))

;;; Matrix operations
;;; -----------------

(defn mul
  "Non-destructive element-wise multiplication of an ndarray with a scalar or other ndarray."
  [^INDArray A a]
  (.mul A a))

(defn mul!
  "Destructive element-wise multiplication of an ndarray with a scalar or other ndarray."
  [^INDArray A a]
  (.muli A a))

(defn mmul
  "Non-destructive iner/outer product of two ndarrays."
  [^INDArray A ^INDArray B]
  (.mmul A B))

(defn mmul!
  "Destructive iner/outer product of two ndarrays."
  [^INDArray A ^INDArray B]
  (.mmuli A B))

(defn add
  "Non-destructive addition of ndarray with scalar or other ndarray"
  [^INDArray A B]
  (.add A B))

(defn add!
  "Destructive addition of ndarray with scalar or other ndarray"
  [^INDArray A B]
  (.addi A B))

(defn sub
  "Non-destructive subtraction of ndarray with scalar or other ndarray"
  [^INDArray A B]
  (.sub A B))

(defn sub!
  "Destructive subtraction of ndarray with scalar or other ndarray"
  [^INDArray A B]
  (.subi A B))

(comment 
  
  (def a (nd4clj.linalg.factory.nd4j/create [1 2 3 4] [2 2]))
  (data a)
  
  )
