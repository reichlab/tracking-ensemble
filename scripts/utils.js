// Common utilities for js scripts

const fs = require('fs-extra')
const yaml = require('js-yaml')
const path = require('path')
const leftPad = require('left-pad')

const writeLines = (lines, filePath) => {
  fs.writeFileSync(filePath, lines.join('\n'))
}

/**
 * Write array to gzipped numpy txt format
 */
const npSaveTxt = (array, filePath) => {
  let lines = []
  for (let row of array) {
    if (Array.isArray(row)) {
      // This is a matrix
      lines.push(row.map(it => Number.parseFloat(it).toExponential()).join(' '))
    } else {
      // This is a vector
      lines.push(Number.parseFloat(row).toExponential())
    }
  }
  writeLines(lines, filePath)
}

/**
 * Return a list of paths to model directories
 */
const getModels = (rootDir) => {
  let modelsDir = path.join(rootDir, 'model-forecasts/component-models')
  return fs.readdirSync(modelsDir)
    .map(it => path.join(modelsDir, it))
    .filter(it => fs.statSync(it).isDirectory())
    .filter(it => fs.existsSync(path.join(it, 'metadata.txt')))
    .map(it => new Model(it))
}

class Model {
  constructor (modelDir) {
    this.dir = modelDir
    this.meta = yaml.safeLoad(fs.readFileSync(path.join(this.dir, 'metadata.txt'), 'utf8'))
    this.id = `${this.meta.team_name}-${this.meta.model_abbr}`
  }

  get csvs () {
    return fs.readdirSync(this.dir)
      .filter(it => it.endsWith('csv'))
      .map(fileName => path.join(this.dir, fileName))
  }

  getCsvFor (epiweek) {
    let csvs = this.csvs
    return csvs[csvs.map(getCsvEpiweek).findIndex(it => it === epiweek)]
  }
}

/**
 * Return timing information about the csv
 */
const getCsvEpiweek = csvFile => {
  let baseName = path.basename(csvFile)
  let [week, year, ] = baseName.split('-')
  return `${year}${leftPad(week.slice(2), 2)}`
}

module.exports.getModels = getModels
module.exports.getCsvEpiweek = getCsvEpiweek
module.exports.writeLines = writeLines
module.exports.Model = Model
module.exports.npSaveTxt = npSaveTxt
