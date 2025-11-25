import type { LPProblem, StandardForm, SlackForm } from './types';

function formatTerm(coef: number, varName: string, isFirst: boolean): string {
  if (coef === 0) return '';
  const sign = coef > 0 ? (isFirst ? '' : ' + ') : ' - ';
  const absCoef = Math.abs(coef);
  const coefStr = absCoef === 1 ? '' : absCoef.toString();
  return `${sign}${coefStr}${varName}`;
}

function formatObjective(coefficients: number[], variableNames: string[]): string {
  return coefficients
    .map((c, i) => formatTerm(c, variableNames[i], i === 0))
    .filter(t => t !== '')
    .join('');
}

function formatConstraintCompact(
  coefficients: number[],
  operator: string,
  rhs: number,
  variableNames: string[]
): string {
  const lhs = coefficients
    .map((c, i) => formatTerm(c, variableNames[i], i === 0))
    .filter(t => t !== '')
    .join('');
  return `${lhs} ${operator} ${rhs}`;
}

export function formatCompact(problem: LPProblem | StandardForm | SlackForm): string {
  const objective = problem.objective === 'maximize' ? '\\text{maximize}' : '\\text{minimize}';
  const objFunc = formatObjective(problem.objectiveCoefficients, problem.variableNames);

  const constraintLines = problem.constraints
    .map(c => formatConstraintCompact(c.coefficients, c.operator, c.rhs, problem.variableNames))
    .map((constraint, i) => i === 0 ? constraint : `& ${constraint}`)
    .join(' \\\\ ');

  return `\\begin{aligned}
${objective} \\quad & ${objFunc} \\\\
\\text{subject to} \\quad & ${constraintLines}
\\end{aligned}`;
}

export function formatFull(problem: LPProblem | StandardForm | SlackForm): string {
  const objective = problem.objective === 'maximize' ? '\\text{maximize}' : '\\text{minimize}';
  const numVars = problem.variableNames.length;

  return `\\begin{aligned}
${objective} \\quad & c^T x = \\begin{bmatrix} ${problem.objectiveCoefficients.join(' \\\\ ')} \\end{bmatrix}^T \\begin{bmatrix} ${problem.variableNames.join(' \\\\ ')} \\end{bmatrix} \\\\
\\text{subject to} \\quad & \\begin{bmatrix} ${problem.constraints.map(c => c.coefficients.slice(0, numVars).join(' & ')).join(' \\\\ ')} \\end{bmatrix} \\begin{bmatrix} ${problem.variableNames.join(' \\\\ ')} \\end{bmatrix}
\\begin{bmatrix} ${problem.constraints.map(c => c.operator).join(' \\\\ ')} \\end{bmatrix} \\begin{bmatrix} ${problem.constraints.map(c => c.rhs).join(' \\\\ ')} \\end{bmatrix}
\\end{aligned}`;
}
