# Linear Programming Equation Forms

An interactive web application for visualizing linear programming problems in different forms.

## Features

- **Dynamic Constraint Management**: Add and remove constraints interactively
- **Multiple Forms**: Switch between Given, Standard, and Slack forms
- **Display Modes**: Toggle between compact notation and full matrix representation
- **Real-time Conversion**: Automatically converts between LP forms
- **Mathematical Rendering**: Uses KaTeX for beautiful mathematical notation
- **Responsive Design**: Clean pastel theme with Tailwind CSS

## Forms Explained

### Given Form

The problem as originally stated with any combination of:

- Maximize or minimize objective
- `≤`, `≥`, or `=` constraints
- Any variable domains

### Standard Form

Normalized representation:

- All inequalities point in the same direction
- All variables are non-negative
- Consistent with the objective type

### Slack Form

Canonical form with slack variables:

- All constraints are equalities
- Slack variables `s_i` added for `≤` constraints
- Surplus variables `e_i` added for `≥` constraints
- All variables non-negative

## Development

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Technology Stack

- **Framework**: Svelte 5 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS + SCSS
- **Math Rendering**: KaTeX
- **Package Manager**: pnpm

## Project Structure

```
src/
├── lib/
│   ├── types.ts              # TypeScript type definitions
│   ├── conversions.ts        # LP form conversion logic
│   ├── latex.ts              # LaTeX formatting utilities
│   ├── MathDisplay.svelte    # KaTeX rendering component
│   └── ConstraintEditor.svelte  # Constraint input component
├── styles/
│   ├── app.scss              # Main stylesheet entry
│   ├── _tailwind.scss        # Tailwind imports
│   └── _base.scss            # Base styles
├── App.svelte                # Main application component
└── main.ts                   # Application entry point
```

## Usage

1. Set the number of variables
2. Choose objective type (maximize/minimize)
3. Enter objective coefficients
4. Add constraints with coefficients, operators, and right-hand sides
5. Switch between form types to see conversions
6. Toggle between compact and full matrix display modes
