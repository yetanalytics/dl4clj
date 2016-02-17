(ns ^{:doc "
A DataSetIterator for use in the graves-lstm-char-modelling-example

@author Joachim De Beule, based on Alex Black's java implementation (see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java)
"}
  dl4clj.examples.rnn.lr-character-iterator
  (:require [dl4clj.examples.example-utils :refer (shakespeare index-map)])
  (:import [java.io File]
           [java.io IOException]
           [java.nio.charset Charset]
           [java.nio.file Files]
           [java.util Arrays]
           [java.util HashMap]
           [java.util LinkedList]
           [java.util List]
           [java.util Map]
           [java.util NoSuchElementException]
           [java.util Random]
           
           [org.apache.commons.io FileUtils]

           [org.deeplearning4j.datasets.iterator DataSetIterator]
           
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.indexing NDArrayIndex]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset.api DataSetPreProcessor]
           [org.nd4j.linalg.factory Nd4j]))

(defrecord LRCharDataSetIterator [valid-chars char-to-idx ^"[C" input-chars segment-length n-segments batch-size char-pointer max-char-pointer]
  DataSetIterator
  (batch [_] batch-size)
  (cursor [_] (mod @char-pointer segment-length)) 
  (inputColumns [_] (count valid-chars))
  (next [this] (.next this batch-size))
  (next [this n]
    (when  (> (+ @char-pointer (* n segment-length)) max-char-pointer)
      (throw (NoSuchElementException.)))
    ;; Allocate space:
    (let [^INDArray input (Nd4j/zeros (int-array [n (count valid-chars) segment-length]))
          ^INDArray labels (Nd4j/zeros (int-array [n (count valid-chars) segment-length]))]
      (dotimes [i n]
        (let [start-idx @char-pointer
              end-idx (swap! char-pointer + segment-length)]
          (.putScalar input (int-array [i (char-to-idx (aget input-chars start-idx)) 0]) 1.0)
          (doseq [c (range 1 segment-length)]
            (let [char-idx (char-to-idx (aget input-chars (+ start-idx c)))]
              (.putScalar input (int-array [i char-idx c]) 1.0)
              (.putScalar labels(int-array [i char-idx (dec c)]) 1.0)))
          (.putScalar labels (int-array [i 
                                         (char-to-idx (aget input-chars end-idx)) 
                                         (dec segment-length)]) 1.0)))
          (DataSet. input labels)))
  (hasNext [this]
    (<= @char-pointer (- max-char-pointer (* batch-size segment-length))))
  (numExamples [this] n-segments)
  (reset [this] (reset! char-pointer 0))
  (totalExamples [this] n-segments) 
  (totalOutcomes [this] (count valid-chars)))

(defmethod print-method LRCharDataSetIterator [iter ^java.io.Writer w]
  (.write w (str "#LRCharDataSetIterator["
                 "n-segments=" (:n-segments iter) 
                 ",batch-size=" (:batch-size iter)
                 ",char-pointer=" @(:char-pointer iter)
                 "]")))

;; (defmethod print-dup LRCharDataSetIterator [iter out]
;;   (.write out (prn iter)))

(defn lr-character-iterator 
  "Reifies a Datasetiterator iterating over text segments in a string from left to right."
  ([string]
   (lr-character-iterator string {}))
  ([^String string {:keys [batch-size ;; number of text segments per batch
                          segment-length ;; number of characters per text segment
                          n-segments ;; number of segments to iterate. Leave unspecified to return the maximum number of segments
                          valid-chars] ;; set of allowed characters. Leave unspecified to allow all characters.
                   :or {batch-size 100
                        segment-length 100}
                   :as opts}]
  (let [valid-chars (into #{} (or valid-chars string))
        chars (char-array (if valid-chars (filter valid-chars string) string))
        max-char-pointer (dec (count chars))
        max-segments (Math/floorDiv max-char-pointer segment-length)]
    (when (and n-segments (> n-segments max-segments)) 
      (throw (IllegalArgumentException. (str "n-segments exceeds number of available segments " max-segments))))
    (when (and n-segments (not (zero? (mod n-segments batch-size))))
      (throw (IllegalArgumentException. (str "n-segments must be a multiple of batch-size"))))
    (println "data has" (count string) "characters," (count valid-chars) "unique.")
    (LRCharDataSetIterator. valid-chars 
                            (index-map valid-chars) 
                            chars 
                            segment-length 
                            (or n-segments max-segments)
                            batch-size 
                            (atom 0)
                            max-char-pointer))))

(defn- char-indices [example ^INDArray features-array]
  (for [pos (range (first (.shape features-array)))]
    (for [fi (range (second (.shape features-array)))]
      (first (.getScalar features-array (int-array [example fi pos]))))))

(defn- binary-feature->char-idx [bf]
  (let [idx (remove #(zero? (first %)) (map vector bf (range)))]
    ;; (assert (= 1 (count idx)))
    (second (first idx))))

(defn- batch-examples [^DataSet batch idx-to-char]
  (for [example (range (.numExamples batch))]
    (let [features-array (.getFeatures batch)
          labels-array (.getLabels batch)]
      {:input (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                              (char-indices example features-array)))
       :output (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                               (char-indices example labels-array)))})))

(defn example-seq
  "Note: NOT thread safe!"
  [^LRCharDataSetIterator iter]
  (.reset iter)
  (let [idx-to-char (zipmap (map (:char-to-idx iter)
                                 (:valid-chars iter))
                            (:valid-chars iter))]
    (mapcat #(batch-examples % idx-to-char) 
            (iterator-seq iter))))



(comment

  (def lr-shakespeare-iterator (lr-character-iterator (shakespeare) {}))

  (:n-segments lr-shakespeare-iterator)
  ;; => 55898

  (take 10 (example-seq lr-shakespeare-iterator))
  ;; => ...
  
  )
