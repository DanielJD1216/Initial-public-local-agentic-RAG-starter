import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium transition-all duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow-[0_14px_32px_rgba(8,112,148,0.26)] hover:-translate-y-0.5 hover:bg-primary/95 active:translate-y-0",
        secondary:
          "border border-white/70 bg-secondary/82 text-secondary-foreground shadow-[0_10px_26px_rgba(15,23,42,0.06)] hover:-translate-y-0.5 hover:bg-secondary/96 active:translate-y-0",
        outline:
          "border border-border/90 bg-white/60 shadow-[0_10px_26px_rgba(15,23,42,0.04)] hover:-translate-y-0.5 hover:bg-accent hover:text-accent-foreground active:translate-y-0",
        ghost: "hover:bg-accent/80 hover:text-accent-foreground",
      },
      size: {
        default: "h-11 px-4 py-2",
        sm: "h-9 rounded-xl px-3 text-xs",
        lg: "h-12 rounded-xl px-6",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
