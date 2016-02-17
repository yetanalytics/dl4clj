(ns ^{:doc "
A DataSetIterator for use in the graves-lstm-char-modelling-example
@author Joachim De Beule, based on Alex Black's java implementation (see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java)
"}
  dl4clj.examples.rnn.character-iterator
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
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset.api DataSetPreProcessor]
           [org.nd4j.linalg.factory Nd4j]))

;; (remove-ns 'dl4clj.examples.rnn.character-iterator)


(def ^:const +minimal-character-set+
  (let [valid-chars (concat (for [c (range (int \a) (inc (int \z)))] (char c))
                            (for [c (range (int \A) (inc (int \Z)))] (char c))
                            (for [c (range (int \0) (inc (int \9)))] (char c))
                            [\!, \&, \(, \), \?, \-, \\, \", ,, \., \:, \;, \space , \newline, \tab])]
    (into #{} valid-chars)))


(def ^:const +default-character-set+
  (let [valid-chars (concat +minimal-character-set+
                            [\@, \#, \$, \%, \^, \*, \{, \}, \[, \], \/, \+, \_, \\\, \|, \<, \>])]
    (into #{} valid-chars)))

(defn- file-characters [path valid-characters cnt]
  (when-not (.exists (clojure.java.io/as-file path))
    (throw  (IOException. (str "Could not access file (does not exist): " path))))
  (char-array (take cnt
                    (filter valid-characters
                            (slurp (clojure.java.io/as-file path))))))

(defn index-map 
  "Utility function to make an map from elements in a collection to indices"
  [col]
  (zipmap col (range)))

(defn character-iterator 
  "Reifies a Datasetiterator iterating over input/output text segments as character ndarrays."
  [path opts]
  (let [{:keys [batch-size         ;; number of text segments per batch
                chars-per-segment  ;; number of characters per text segment
                max-segments       ;; maximum number of segments to return
                start-characters ;; set of characters that can start a text segment. Set scan-limit to zero if any character can start a segment.
                scan-limit ;; limit to scan for start characters before giving up and returning a segment starting with a non-start character.
                valid-characters] ;; set of valid characters. Any other characters are dismissed.
         :or {batch-size 32
              chars-per-segment 100
              max-segments (* 50 32)
              start-characters #{\newline}
              scan-limit 200
              valid-characters +default-character-set+}} opts
              char-to-idx-map (index-map valid-characters)
              segment-count (atom 0)]
    (when-not (zero? (mod max-segments batch-size)) 
      (throw (IllegalArgumentException. "max-segments must be a multiple of batch-size")))
    (let [file-characters ^"[C" (file-characters path 
                                                 (into #{} (get opts :valid-characters +default-character-set+))
                                                 (* max-segments chars-per-segment))
          max-start-idx (dec (- (count file-characters) chars-per-segment))
          random-start-idx (fn []
                             (loop [idx (rand-int max-start-idx)
                                    scan-count 0]
                               (if (and (< scan-count scan-limit)
                                        (>= idx 1) 
                                        (not (contains? start-characters (aget file-characters (dec idx)))))
                                 (recur (dec idx) (inc scan-count))
                                 idx)))]
      (when (>= chars-per-segment (count file-characters))
        (throw ( IllegalArgumentException. (str "chars-per-segment=" chars-per-segment
                                                " cannot exceed number of valid characters in file (" (count file-characters))")")))
      (reify DataSetIterator
        (batch [_] batch-size)
        (cursor [_] @segment-count) 
        (inputColumns [_] (count valid-characters))
        (next [this] (.next this batch-size))
        (next [this n-segments]
          (when  (> (+ @segment-count n-segments) max-segments)
            (throw (NoSuchElementException.)))
          ;; Allocate space:
          (let [^INDArray input (Nd4j/zeros (int-array [n-segments (count valid-characters) chars-per-segment]))
                ^INDArray labels (Nd4j/zeros (int-array [n-segments (count valid-characters) chars-per-segment]))] ;
            ;; Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
            ;; of the file in the same minibatch
            (doseq [i (range n-segments)]
              (let [start-idx (random-start-idx)
                    end-idx (+ start-idx chars-per-segment)]
                (loop [c 0 
                       j (inc start-idx)
                       curr-char-idx (get char-to-idx-map (aget file-characters start-idx))]
                  (let [next-char-idx (get char-to-idx-map (aget file-characters j))]
                    (.putScalar input (int-array [i curr-char-idx c]) 1.0)
                    (.putScalar labels (int-array [i next-char-idx c]) 1.0)
                    (when (< j end-idx) (recur (inc c) (inc j) next-char-idx))))))
            (swap! segment-count #(+ % n-segments))
            (DataSet. input labels)))
        (hasNext [this]
          (<= (+ @segment-count batch-size)
              max-segments))
        (numExamples [this] max-segments)
        (reset [this] (reset! segment-count 0))
        (totalExamples [this] max-segments) 
        (totalOutcomes [this] (count valid-characters))))))

(defn shakespeare-iterator
  "Downloads Shakespeare training data and stores it locally (temp directory). Then sets up and
  returns a DataSetIterator that does vectorization based on the text.
  options:   
     batch-size:            Number of examples (text segments) in each training batch 
     chars-per-segment:     Number of characters in each text segment.  
     max-segments:          Total number of egments (should be a multiple of batch-size)
"
  [{:keys [batch-size chars-per-segment max-segments valid-characters]
    :or {:batch-size 32
         :chars-per-segment 100 
         :max-segments (* 50 32)
         :valid-characters +default-character-set+}
    :as opts}]
  ;; The Complete Works of William Shakespeare
  ;; 5.3MB file in UTF-8 Encoding, ~5.4 million characters
  ;; https://www.gutenberg.org/ebooks/100
  (let [url "https://s3.amazonaws.com/dl4j-distribution/pg100.txt"
        temp-dir  (System/getProperty "java.io.tmpdir")
        file-location  (str temp-dir "/Shakespeare.txt")
        f (clojure.java.io/as-file file-location)]
    (when-not (.exists f)
      (do (FileUtils/copyURLToFile (clojure.java.io/as-url url) f)
          (println "File downloaded to " (.getAbsolutePath f))))
    (when-not (.exists f) 
      (throw (IOException. (str "File does not exist: " file-location))))
    (character-iterator file-location opts)))

;; (ns ^{:doc "
;; A DataSetIterator for use in the graves-lstm-char-modelling-example

;; @author Joachim De Beule, based on Alex Black's java implementation (see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java)
;; "}
;;   dl4clj.examples.rnn.character-iterator
;;   (:require [dl4clj.examples.example-utils :refer (shakespeare index-map)])
;;   (:import [java.io File]
;;            [java.io IOException]
;;            [java.nio.charset Charset]
;;            [java.nio.file Files]
;;            [java.util Arrays]
;;            [java.util HashMap]
;;            [java.util LinkedList]
;;            [java.util List]
;;            [java.util Map]
;;            [java.util NoSuchElementException]
;;            [java.util Random]
           
;;            [org.apache.commons.io FileUtils]

;;            [org.deeplearning4j.datasets.iterator DataSetIterator]

;;            [org.nd4j.linalg.api.ndarray INDArray]
;;            [org.nd4j.linalg.indexing NDArrayIndex]
;;            [org.nd4j.linalg.dataset DataSet]
;;            [org.nd4j.linalg.dataset.api DataSetPreProcessor]
;;            [org.nd4j.linalg.factory Nd4j]))

;; (defn character-iterator 
;;   "Reifies a Datasetiterator iterating over text segments in a string."
;;   [string {:keys [batch-size      ;; number of text segments per batch
;;                   segment-length ;; number of characters per text segment
;;                   n-segments ;; number of segments to iterate. Leave unspecified to return the maximum number of segments
;;                   valid-characters] ;; set of allowed characters. Leave unspecified to allow all characters.
;;            :or {batch-size 100
;;                 segment-length 100}
;;            :as opts}]
;;   (let [chars (char-array (if valid-characters (filter valid-characters string) string))
;;         valid-characters (or valid-characters (into #{} string))
;;         char-to-idx-map (index-map valid-characters)
;;         segment-count (atom 0)
;;         max-segments (mod (count string) segment-length) 
;;         max-batches (mod (count string) (* segment-length batch-size))]
;;     (when (and n-segments (> n-segments max-segments)) 
;;       (throw (IllegalArgumentException. (str "n-segments exceeds number of available segments " max-segments))))
;;     (reify DataSetIterator
;;       (batch [_] batch-size)
;;       (cursor [_] @segment-count) 
;;       (inputColumns [_] (count valid-characters))
;;       (next [this] (.next this batch-size))
;;       (next [this n-segments]
;;         (when  (> (+ @segment-count n-segments) max-segments)
;;           (throw (NoSuchElementException.)))
;;         ;; Allocate space:
;;         (let [^INDArray input (Nd4j/zeros (int-array [n-segments (count valid-characters) segment-length]))
;;               ^INDArray labels (Nd4j/zeros (int-array [n-segments (count valid-characters) segment-length]))]
;;           (dotimes [i n-segments]
;;             (let [start-idx (random-start-idx)
;;                   end-idx (+ start-idx chars-per-segment)]
;;               (loop [c 0 
;;                      j (inc start-idx)
;;                      curr-char-idx (get char-to-idx-map (aget file-characters start-idx))]
;;                 (let [next-char-idx (get char-to-idx-map (aget file-characters j))]
;;                   (.putScalar input (int-array [i curr-char-idx c]) 1.0)
;;                   (.putScalar labels (int-array [i next-char-idx c]) 1.0)
;;                   (when (< j end-idx) (recur (inc c) (inc j) next-char-idx))))))
;;           (swap! segment-count #(+ % n-segments))
;;           (DataSet. input labels)))
;;       (hasNext [this]
;;         (<= (+ @segment-count batch-size)
;;             max-segments))
;;       (numExamples [this] max-segments)
;;       (reset [this] (reset! segment-count 0))
;;       (totalExamples [this] max-segments) 
;;       (totalOutcomes [this] (count valid-characters)))))



;; (comment

;;   )
    
