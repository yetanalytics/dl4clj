(ns ^{:doc "
A DataSetIterator for use in the graves-lstm-char-modelling-example

@author Joachim De Beule, based on Alex Black's java implementation (see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java)
"}
  dl4clj.examples.rnn.character-iterator
  (:require [nd4clj.linalg.factory.nd4j :refer (zeros)]
            [dl4clj.examples.example-utils :refer (index-map)]
            [nd4clj.linalg.api.ndarray.indarray :refer (put-scalar get-scalar shape)]
            [nd4clj.linalg.dataset.api.data-set :refer :all]
            [nd4clj.linalg.dataset.data-set :refer (data-set)])
  (:import [java.util NoSuchElementException]
           [java.util Random]
           [org.deeplearning4j.datasets.iterator DataSetIterator]))

;; (remove-ns 'dl4clj.examples.rnn.character-iterator)


(def ^:const +minimal-character-set+
  (let [valid-chars (concat (for [c (range (int \a) (inc (int \z)))] (char c))
                            (for [c (range (int \A) (inc (int \Z)))] (char c))
                            (for [c (range (int \0) (inc (int \9)))] (char c))
                            [\!, \&, \(, \), \?, \-, \\, \", ,, \., \:, \;, \space , \newline, \tab])]
    (into #{} valid-chars)))

(def ^:const +default-character-set+
  (let [valid-chars (concat +minimal-character-set+
                            [\@, \#, \$, \%, \^, \*, \{, \}, \[, \], \/, \+, \_, \\, \|, \<, \>])]
    (into #{} valid-chars)))

(defn- file-characters [path valid-characters]
  (when-not (.exists (clojure.java.io/as-file path))
    (throw  (IOException. (str "Could not access file (does not exist): " path))))
  (char-array (filter (into #{} valid-characters)
                      (slurp (clojure.java.io/as-file path)))))

(defn character-iterator 
  "Reifies a Datasetiterator iterating over input/output text segments as character ndarrays."
  [path {:keys [batch-size
                example-length
                num-examples-to-fetch
                valid-characters
                rng
                always-start-at-newline?]
         :or {valid-characters +minimal-character-set+}
         :as opts}]
  (let [char-to-idx-map (index-map valid-characters)
        num-characters (count valid-characters)
        file-characters ^"[C" (file-characters path valid-characters)
        examples-so-far (atom 0)]
    (reify DataSetIterator
      (hasNext [this] (<= (+ @examples-so-far batch-size) num-examples-to-fetch ))
      (next [this] (.next this batch-size))
      (next [this num]
        (when  (> @examples-so-far num-examples-to-fetch)
          (throw (NoSuchElementException.)))
        ;; Allocate space:
        (let [input (zeros [num num-characters example-length])
              labels (zeros [num num-characters example-length])
              max-start-idx (- (count file-characters) example-length)]
          ;; Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
          ;; of the file in the same minibatch
          (dotimes [i num]
            (let [start-idx (loop [i (int (* (.nextDouble ^Random rng) max-start-idx))
                                   scan 0]
                              (if (and (< scan 200) (not= (aget file-characters (dec i)) \newline) always-start-at-newline? (>= i 1))
                                (recur (dec i) (inc scan))
                                i))]
              (dotimes [j example-length]
                (let [file-idx (+ start-idx j)
                      current-char (aget file-characters file-idx)
                      next-char (aget file-characters (inc file-idx))]
                  (put-scalar input [i (char-to-idx-map current-char) j] 1.0)
                  (put-scalar labels [i (char-to-idx-map next-char) j] 1.0)))
              (put-scalar labels [i (char-to-idx-map (aget file-characters (+ start-idx example-length))) (dec example-length)] 1.0)))
          (swap! examples-so-far #(+ % num))
          (data-set input labels)))
      (totalExamples [this] num-examples-to-fetch) 
      (inputColumns [_] num-characters)
      (totalOutcomes [this] num-characters)
      (reset [this] (reset! examples-so-far 0))                
      (batch [_] batch-size)        
      (cursor [_] @examples-so-far) 
      (numExamples [this] num-examples-to-fetch))))


(comment


  ;;; some (inneficient) code for inspecting the examples in an lr-character-iterator

  (defn- char-indices [example features-array]
    (for [pos (range (first (shape features-array)))]
      (for [fi (range (second (shape features-array)))]
        (first (get-scalar features-array [example fi pos])))))

  (defn- binary-feature->char-idx [bf]
    (let [idx (remove #(zero? (first %)) (map vector bf (range)))]
      (second (first idx))))

  (defn- batch-examples [batch idx-to-char]
    (for [example (range (num-examples batch))]
      (let [features-array (get-features batch)
            labels-array (get-labels batch)]
        {:input (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                                (char-indices example features-array)))
         :output (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                                 (char-indices example labels-array)))})))

  (defn example-seq
    "Note: NOT thread safe!"
    [iter]
    (reset iter)
    (let [idx-to-char (zipmap (map (:char-to-idx iter)
                                   (:valid-chars iter))
                              (:valid-chars iter))]
      (mapcat #(batch-examples % idx-to-char) 
              (iterator-seq iter))))

  )
