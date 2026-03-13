import * as TabsPrimitive from "@radix-ui/react-tabs";

import { cn } from "@/lib/utils";

const Tabs = TabsPrimitive.Root;

const TabsList = ({ className, ...props }: TabsPrimitive.TabsListProps) => (
  <TabsPrimitive.List
    className={cn(
      "inline-flex h-11 items-center rounded-2xl border border-white/70 bg-secondary/70 p-1 text-muted-foreground shadow-[0_10px_26px_rgba(15,23,42,0.05)]",
      className,
    )}
    {...props}
  />
);

const TabsTrigger = ({ className, ...props }: TabsPrimitive.TabsTriggerProps) => (
  <TabsPrimitive.Trigger
    className={cn(
      "inline-flex items-center justify-center rounded-xl px-3 py-1.5 text-sm font-medium transition-all duration-200 data-[state=active]:bg-background/95 data-[state=active]:text-foreground data-[state=active]:shadow-[0_10px_18px_rgba(15,23,42,0.08)]",
      className,
    )}
    {...props}
  />
);

const TabsContent = ({ className, ...props }: TabsPrimitive.TabsContentProps) => (
  <TabsPrimitive.Content className={cn("mt-4 outline-none", className)} {...props} />
);

export { Tabs, TabsContent, TabsList, TabsTrigger };
